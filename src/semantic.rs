use std::collections::{HashMap, HashSet};
use crate::ast::*;

#[derive(Debug, Clone)]
struct VarInfo {
    ty: Type,
    mutable: bool,
}

struct Scope {
    vars: HashMap<String, VarInfo>,
}

pub struct SemanticAnalyzer {
    scopes: Vec<Scope>,
    functions: HashMap<String, (Vec<Type>, Type)>,
    structs: HashMap<String, Vec<(String, Type)>>,  // name -> [(field, type)]
    /// Names that came from a builtin table rather than from the program.
    /// Only builtins are pool-polymorphic; a user `@cpu fn(tensor@cpu)` stays
    /// a promise about where the work happens.
    builtins: HashSet<String>,
    in_gpu_block: bool,
    loop_depth: usize,
    errors: Vec<String>,
}


/// Signatures for the MLX tensor backend.
///
/// Kept as one table that both the checker and the LLVM codegen consult, so a
/// builtin cannot exist in one compiler phase but not the other. Handles are
/// `tensor@cpu`: a host-side handle to a lazily-built device graph. Typing them
/// `tensor@gpu` would make every use outside `gpu { }` a type error.
pub fn mlx_builtins(m: &mut HashMap<String, (Vec<Type>, Type)>) {
    // NOTE: codegen emits LLVM `declare` lines by walking this same map, so
    // every entry here is automatically linkable from generated IR.
    // ── MLX tensor backend ────────────────────────────────────────────────
    // Mirrors register_builtins() in src/semantic/checker.ionic. Handles are
    // `tensor@cpu`: a host-side handle to a lazily-built device graph. Typing
    // them `tensor@gpu` would make every use outside `gpu { }` a type error.
    #[allow(unused)]
    let f   = Type::Float64;
    let t   = || Type::Tensor(Box::new(HwTarget::Cpu));
    let i   = Type::Int64;
    let a   = Type::Array(Box::new(Type::Int64));
    let af  = Type::Array(Box::new(Type::Float64));
    let s_  = Type::Str;

    // Construction (shape arg is an [int64] array; dtype is an int)
    m.insert("mlx_zeros".to_string(), (vec![a.clone(), i.clone()], t()));
    m.insert("mlx_ones".to_string(),  (vec![a.clone(), i.clone()], t()));
    // dtype cast / keepdims reductions / axis-wise softmax
    for name in ["mlx_astype", "mlx_softmax", "mlx_log_softmax"] {
        m.insert(name.to_string(), (vec![t(), i.clone()], t()));
    }
    for name in ["mlx_sum", "mlx_mean", "mlx_max", "mlx_min"] {
        m.insert(name.to_string(), (vec![t(), i.clone()], t()));
    }
    for name in ["mlx_add", "mlx_subtract", "mlx_multiply", "mlx_divide",
                 "mlx_power", "mlx_maximum", "mlx_minimum", "mlx_matmul"] {
        m.insert(name.to_string(), (vec![t(), t()], t()));
    }
    for name in ["mlx_negative", "mlx_square", "mlx_sqrt", "mlx_exp",
                 "mlx_log", "mlx_abs", "mlx_tanh", "mlx_erf", "mlx_contiguous"] {
        m.insert(name.to_string(), (vec![t()], t()));
    }
    m.insert("mlx_full".to_string(),    (vec![a.clone(), f.clone(), i.clone()], t()));
    m.insert("mlx_arange".to_string(),  (vec![f.clone(), f.clone(), f.clone(), i.clone()], t()));
    m.insert("mlx_eye".to_string(),     (vec![i.clone(), i.clone(), i.clone()], t()));
    m.insert("mlx_random_normal".to_string(),  (vec![a.clone(), f.clone(), f.clone(), i.clone()], t()));
    m.insert("mlx_random_uniform".to_string(), (vec![a.clone(), f.clone(), f.clone(), i.clone()], t()));
    m.insert("mlx_from_array".to_string(), (vec![af.clone(), a.clone(), i.clone()], t()));
    // Shape manipulation: (tensor, [int64] axes/shape) -> tensor
    m.insert("mlx_reshape".to_string(),   (vec![t(), a.clone()], t()));
    m.insert("mlx_transpose".to_string(), (vec![t(), a.clone()], t()));
    // Lifetime / materialisation
    m.insert("mlx_free".to_string(),   (vec![t()], Type::Void));
    m.insert("mlx_eval".to_string(),   (vec![t()], i.clone()));
    m.insert("mlx_item".to_string(),   (vec![t()], f.clone()));
    m.insert("mlx_to_array".to_string(), (vec![t()], af.clone()));
    m.insert("mlx_shape".to_string(),  (vec![t()], a.clone()));
    // Inspection
    for name in ["mlx_ndim", "mlx_size", "mlx_dtype"] {
        m.insert(name.to_string(), (vec![t()], i.clone()));
    }
    m.insert("mlx_dim".to_string(), (vec![t(), i.clone()], i.clone()));
    // Pool introspection (mirrors the checker's registration in
    // src/semantic/checker.ionic): which pool a tensor lives in, and how many
    // bytes each pool holds. Both return int64.
    m.insert("mlx_pool".to_string(),     (vec![t()], i.clone()));
    m.insert("mlx_pool_bytes".to_string(), (vec![i.clone()], i.clone()));
    m.insert("mlx_pool_count".to_string(), (vec![], i.clone()));
    m.insert("mlx_pools_unified".to_string(), (vec![], i.clone()));
    // Axis reductions
    for name in ["mlx_sum_axis", "mlx_mean_axis", "mlx_argmax"] {
        m.insert(name.to_string(), (vec![t(), i.clone(), i.clone()], t()));
    }
    // Slicing / gather
    m.insert("mlx_slice".to_string(),       (vec![t(), i.clone(), i.clone(), i.clone()], t()));
    m.insert("mlx_take".to_string(),        (vec![t(), a.clone(), i.clone()], t()));
    m.insert("mlx_concatenate".to_string(), (vec![t(), t(), i.clone()], t()));
    // NN primitives
    m.insert("mlx_sdpa".to_string(),       (vec![t(), t(), t(), f.clone(), i.clone()], t()));
    m.insert("mlx_rms_norm".to_string(),   (vec![t(), t(), f.clone()], t()));
    m.insert("mlx_layer_norm".to_string(), (vec![t(), t(), t(), f.clone()], t()));
    m.insert("mlx_rope".to_string(),       (vec![t(), i.clone(), f.clone(), i.clone(), i.clone()], t()));
    // Diagnostics / memory
    for name in ["mlx_available", "mlx_metal_available", "mlx_clear_error",
                 "mlx_active_memory", "mlx_peak_memory", "mlx_reset_peak_memory",
                 "mlx_clear_cache"] {
        m.insert(name.to_string(), (vec![], i.clone()));
    }
    m.insert("mlx_set_memory_limit".to_string(), (vec![i.clone()], i.clone()));
    m.insert("mlx_last_error".to_string(), (vec![], s_.clone()));
    // Safetensors (weight handles are plain int64 — they are not tensors)
    m.insert("mlx_load_safetensors".to_string(), (vec![s_.clone()], i.clone()));
    m.insert("mlx_weights_free".to_string(),     (vec![i.clone()], Type::Void));
    m.insert("mlx_weights_count".to_string(),    (vec![i.clone()], i.clone()));
    m.insert("mlx_weights_at".to_string(),       (vec![i.clone(), i.clone()], t()));
    m.insert("mlx_weights_key_at".to_string(),   (vec![i.clone(), i.clone()], s_.clone()));
    m.insert("mlx_weights_get".to_string(),      (vec![i.clone(), s_.clone()], t()));
    m.insert("mlx_save_safetensors".to_string(), (vec![t(), s_.clone(), s_.clone()], i.clone()));
}

/// Signatures for the non-MLX runtime surface: HTTP client/server, BPE
/// tokenizer, and the TUI primitives.
///
/// Same contract as `mlx_builtins`: one table, read by both the checker and
/// codegen, so the two phases cannot disagree. These all live in
/// src/ionic_model_runtime.c and map to `ionic_*` symbols.
///
/// Getting the *return* type right is not cosmetic here. An unregistered name
/// infers as `Type::Unknown`, and codegen's `print` path treats Unknown as an
/// i64 — so `print(bpe_decode(ids))` would emit `printf("%lld", ptr)` and fail
/// the LLVM verifier. Registering the string returns fixes that at the source.
pub fn runtime_builtins(m: &mut HashMap<String, (Vec<Type>, Type)>) {
    let i  = Type::Int64;
    let s_ = Type::Str;
    let a  = Type::Array(Box::new(Type::Int64));

    // ── HTTP client (returns the raw response, headers + body) ───────────────
    for name in ["http_get", "http_body"] {
        m.insert(name.to_string(), (vec![s_.clone()], s_.clone()));
    }
    m.insert("http_post".to_string(),     (vec![s_.clone(), s_.clone()], s_.clone()));
    m.insert("http_status".to_string(),   (vec![s_.clone()], i.clone()));
    m.insert("http_urlencode".to_string(),(vec![s_.clone()], s_.clone()));

    // ── BPE tokenizer ────────────────────────────────────────────────────────
    m.insert("bpe_load".to_string(),   (vec![s_.clone()], i.clone()));
    m.insert("bpe_encode".to_string(), (vec![s_.clone(), a.clone(), i.clone()], i.clone()));
    m.insert("bpe_decode".to_string(), (vec![a.clone()], s_.clone()));
    for name in ["bpe_bos", "bpe_eos"] {
        m.insert(name.to_string(), (vec![], i.clone()));
    }
    m.insert("arr_push_i64".to_string(), (vec![a.clone(), i.clone()], i.clone()));

    // ── HTTP server ──────────────────────────────────────────────────────────
    // A request is an opaque int64 handle that owns its connection: `accept`
    // returns it, and `respond`/`req_done` take that same handle back.
    m.insert("http_serve".to_string(),    (vec![i.clone()], i.clone()));
    m.insert("http_accept".to_string(),   (vec![i.clone()], i.clone()));
    for name in ["http_method", "http_path", "http_req_body"] {
        m.insert(name.to_string(), (vec![i.clone()], s_.clone()));
    }
    m.insert("http_respond".to_string(),
        (vec![i.clone(), i.clone(), s_.clone(), s_.clone()], i.clone()));
    for name in ["http_req_done", "http_close"] {
        m.insert(name.to_string(), (vec![i.clone()], i.clone()));
    }

    // ── Terminal / TUI ────────────────────────────────────────────────────────
    for name in ["term_enter", "term_exit", "term_flush", "term_size", "term_key"] {
        m.insert(name.to_string(), (vec![], i.clone()));
    }
    // term_write takes the string to emit; term_poll takes a millisecond timeout.
    m.insert("term_write".to_string(), (vec![s_.clone()], i.clone()));
    m.insert("term_poll".to_string(),  (vec![i.clone()],   i.clone()));
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        let mut sa = SemanticAnalyzer {
            scopes: vec![Scope { vars: HashMap::new() }],
            functions: HashMap::new(),
            structs: HashMap::new(),
            builtins: HashSet::new(),
            in_gpu_block: false,
            loop_depth: 0,
            errors: Vec::new(),
        };
        // Built-in functions
        sa.functions.insert("print".to_string(),            (vec![Type::Unknown], Type::Void));
        sa.functions.insert("println".to_string(),          (vec![Type::Unknown], Type::Void));
        sa.functions.insert("exit".to_string(),             (vec![Type::Int64],   Type::Void));
        sa.functions.insert("sqrt".to_string(),             (vec![Type::Float64], Type::Float64));
        sa.functions.insert("abs".to_string(),              (vec![Type::Float64], Type::Float64));
        sa.functions.insert("int64_to_float64".to_string(), (vec![Type::Int64],   Type::Float64));
        sa.functions.insert("float64_to_int64".to_string(), (vec![Type::Float64], Type::Int64));
        // Array builtins
        sa.functions.insert("len".to_string(),      (vec![Type::Unknown], Type::Int64));
        sa.functions.insert("push".to_string(),     (vec![Type::Unknown, Type::Unknown], Type::Void));
        sa.functions.insert("pop".to_string(),      (vec![Type::Unknown], Type::Unknown));
        // String builtins
        sa.functions.insert("str_len".to_string(),    (vec![Type::Str], Type::Int64));
        sa.functions.insert("str_concat".to_string(), (vec![Type::Str, Type::Str], Type::Str));
        sa.functions.insert("str_eq".to_string(),     (vec![Type::Str, Type::Str], Type::Bool));
        sa.functions.insert("str_index".to_string(),  (vec![Type::Str, Type::Int64], Type::Int64));
        sa.functions.insert("int64_to_str".to_string(),(vec![Type::Int64], Type::Str));
        sa.functions.insert("float64_to_str".to_string(),(vec![Type::Float64], Type::Str));
        sa.functions.insert("char_to_str".to_string(), (vec![Type::Int64], Type::Str));
        // System builtins
        sa.functions.insert("get_arg".to_string(),         (vec![Type::Int64], Type::Str));
        sa.functions.insert("cpu_core_count".to_string(),  (vec![],            Type::Int64));
        sa.functions.insert("file_exists".to_string(),     (vec![Type::Str],   Type::Bool));
        sa.functions.insert("system".to_string(),          (vec![Type::Str],   Type::Int64));
        // ML model builtins
        sa.functions.insert("load_model".to_string(),
            (vec![Type::Str], Type::Model(Box::new(HwTarget::Cpu))));
        sa.functions.insert("model_free".to_string(),
            (vec![Type::Model(Box::new(HwTarget::Cpu))], Type::Void));
        sa.functions.insert("piper_forward".to_string(),
            (vec![Type::Model(Box::new(HwTarget::Cpu)), Type::Array(Box::new(Type::Int64)),
                  Type::Float64, Type::Float64, Type::Float64],
             Type::Array(Box::new(Type::Float64))));
        sa.functions.insert("write_wav".to_string(),
            (vec![Type::Str, Type::Array(Box::new(Type::Float64)), Type::Int64, Type::Int64],
             Type::Void));
        sa.functions.insert("fgets_stdin".to_string(),
            (vec![Type::Int64], Type::Str));
        sa.functions.insert("file_write_binary".to_string(),
            (vec![Type::Array(Box::new(Type::Int64)), Type::Str], Type::Int64));
        sa.functions.insert("arr_reset".to_string(),
            (vec![Type::Array(Box::new(Type::Int64))], Type::Int64));

        mlx_builtins(&mut sa.functions);
        runtime_builtins(&mut sa.functions);
        // File I/O builtins
        sa.functions.insert("file_read".to_string(),  (vec![Type::Str], Type::Str));
        sa.functions.insert("file_write".to_string(), (vec![Type::Str, Type::Str], Type::Bool));
        sa.functions.insert("file_open_read".to_string(),  (vec![Type::Str], Type::Int64));
        sa.functions.insert("file_open_write".to_string(), (vec![Type::Str], Type::Int64));
        sa.functions.insert("file_close".to_string(),      (vec![Type::Int64], Type::Void));
        sa.functions.insert("file_read_line".to_string(),  (vec![Type::Int64], Type::Str));
        sa.functions.insert("file_write_line".to_string(), (vec![Type::Int64, Type::Str], Type::Void));
        sa.functions.insert("file_eof".to_string(),        (vec![Type::Int64], Type::Bool));

        // Everything registered so far came from a builtin table, so it is
        // pool-polymorphic. User functions are inserted into `functions` later
        // in analyze() and are deliberately absent from this set.
        sa.builtins = sa.functions.keys().cloned().collect();
        sa
    }

    pub fn analyze(&mut self, program: &Program) -> Vec<String> {
        // Register struct types first
        for s in &program.structs {
            let fields: Vec<(String, Type)> = s.fields.iter()
                .map(|f| (f.name.clone(), f.ty.to_type()))
                .collect();
            self.structs.insert(s.name.clone(), fields);
        }

        // Register all user functions (forward declarations)
        for f in &program.fns {
            let param_types: Vec<Type> = f.params.iter().map(|p| p.ty.to_type()).collect();
            self.functions.insert(f.name.clone(), (param_types, f.ret_ty.to_type()));
        }

        // Top-level lets/muts first so function bodies can reference them
        for stmt in &program.top_level {
            self.analyze_stmt(stmt, &Type::Void);
        }

        for f in &program.fns {
            self.analyze_fn(f);
        }

        self.errors.clone()
    }

    fn analyze_fn(&mut self, f: &FnDef) {
        // If @gpu-annotated, enter gpu block context automatically
        let was_gpu = self.in_gpu_block;
        if let Some(hw) = &f.hw {
            if hw.is_gpu() {
                self.in_gpu_block = true;
            }
        }
        self.push_scope();
        for p in &f.params {
            let pty = p.ty.to_type();
            // Warn if param hw annotation conflicts with function hw annotation
            if let (Some(fn_hw), Some(p_hw)) = (&f.hw, &p.hw) {
                if fn_hw.is_gpu() && p_hw.is_cpu() {
                    self.error(format!(
                        "Function `{}` is @gpu but parameter `{}` is @cpu — remove one annotation",
                        f.name, p.name
                    ));
                } else if fn_hw.is_cpu() && p_hw.is_gpu() {
                    self.error(format!(
                        "Function `{}` is @cpu but parameter `{}` is @gpu — remove one annotation",
                        f.name, p.name
                    ));
                }
            }
            self.declare_var(&p.name, pty, false);
        }
        let ret_ty = f.ret_ty.to_type();
        for stmt in &f.body {
            self.analyze_stmt(stmt, &ret_ty);
        }
        self.pop_scope();
        self.in_gpu_block = was_gpu;
    }

    fn push_scope(&mut self) { self.scopes.push(Scope { vars: HashMap::new() }); }
    fn pop_scope(&mut self)  { self.scopes.pop(); }

    fn declare_var(&mut self, name: &str, ty: Type, mutable: bool) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.vars.insert(name.to_string(), VarInfo { ty, mutable });
        }
    }

    fn lookup_var(&self, name: &str) -> Option<&VarInfo> {
        for scope in self.scopes.iter().rev() {
            if let Some(info) = scope.vars.get(name) {
                return Some(info);
            }
        }
        None
    }

    fn error(&mut self, msg: String) {
        self.errors.push(msg);
    }

    /// Type compatibility, matching the self-hosted checker's `ty_compat`:
    /// unknown on either side is compatible, bool and int64 are the same at
    /// runtime, and otherwise the types must be equal.
    fn ty_compat(got: &Type, want: &Type) -> bool {
        if *want == Type::Unknown || *got == Type::Unknown {
            return true;
        }
        if got == want {
            return true;
        }
        matches!((got, want),
            (Type::Bool, Type::Int64) | (Type::Int64, Type::Bool))
    }

    fn analyze_stmt(&mut self, stmt: &Stmt, ret_ty: &Type) {
        match stmt {
            Stmt::Let { mutable, name, ty, init, hw: _ } => {
                let inferred = self.infer_expr(init);
                let resolved = if let Some(annotation) = ty {
                    let ann_ty = annotation.to_type();
                    if ann_ty != inferred && inferred != Type::Unknown {
                        self.error(format!(
                            "Type mismatch for `{}`: declared `{}` but init is `{}`",
                            name, ann_ty, inferred
                        ));
                    }
                    ann_ty
                } else {
                    inferred
                };
                self.declare_var(name, resolved, *mutable);
            }

            Stmt::Assign { target, value } => {
                let val_ty = self.infer_expr(value);
                match target {
                    AssignTarget::Ident(name) => {
                        let info = self.lookup_var(name).cloned();
                        match info {
                            None => self.error(format!("Undefined variable `{}`", name)),
                            Some(info) => {
                                if !info.mutable {
                                    self.error(format!("Cannot assign to immutable variable `{}`", name));
                                }
                                if !Self::ty_compat(&val_ty, &info.ty) {
                                    self.error(format!(
                                        "Type mismatch in assignment to `{}`: expected `{}`, got `{}`",
                                        name, info.ty, val_ty
                                    ));
                                }
                            }
                        }
                    }
                    AssignTarget::Field(obj, field) => {
                        let obj_ty = self.infer_expr(obj);
                        if let Type::Struct(sname) = &obj_ty {
                            if let Some(fields) = self.structs.get(sname).cloned() {
                                if fields.iter().find(|(f, _)| f == field).is_none() {
                                    self.error(format!("No field `{}` on struct `{}`", field, sname));
                                }
                            }
                        }
                    }
                    AssignTarget::Index(obj, idx) => {
                        self.infer_expr(obj);
                        self.infer_expr(idx);
                    }
                }
            }

            Stmt::ExprStmt(e) => {
                self.infer_expr(e);
                if self.in_gpu_block {
                    self.check_gpu_restrictions(e);
                }
            }

            Stmt::Return(expr) => {
                let actual = expr.as_ref().map(|e| self.infer_expr(e)).unwrap_or(Type::Void);
                if *ret_ty != Type::Void && !Self::ty_compat(&actual, ret_ty) {
                    self.error(format!(
                        "Return type mismatch: expected `{}`, got `{}`", ret_ty, actual
                    ));
                }
            }

            Stmt::If { cond, then_block, else_block } => {
                let ct = self.infer_expr(cond);
                if ct != Type::Bool && ct != Type::Int64 && ct != Type::Unknown {
                    self.error(format!("if condition must be `bool`, got `{}`", ct));
                }
                self.push_scope();
                for s in then_block { self.analyze_stmt(s, ret_ty); }
                self.pop_scope();
                if let Some(eb) = else_block {
                    self.push_scope();
                    for s in eb { self.analyze_stmt(s, ret_ty); }
                    self.pop_scope();
                }
            }

            Stmt::While { cond, body } => {
                let ct = self.infer_expr(cond);
                if ct != Type::Bool && ct != Type::Int64 && ct != Type::Unknown {
                    self.error(format!("while condition must be `bool`, got `{}`", ct));
                }
                self.loop_depth += 1;
                self.push_scope();
                for s in body { self.analyze_stmt(s, ret_ty); }
                self.pop_scope();
                self.loop_depth -= 1;
            }

            Stmt::For { var, start, end, body } => {
                let st = self.infer_expr(start);
                let et = self.infer_expr(end);
                if st != Type::Int64 && st != Type::Unknown {
                    self.error(format!("for range start must be `int64`, got `{}`", st));
                }
                if et != Type::Int64 && et != Type::Unknown {
                    self.error(format!("for range end must be `int64`, got `{}`", et));
                }
                self.loop_depth += 1;
                self.push_scope();
                self.declare_var(var, Type::Int64, false);
                for s in body { self.analyze_stmt(s, ret_ty); }
                self.pop_scope();
                self.loop_depth -= 1;
            }

            Stmt::GpuBlock(body) => {
                let was = self.in_gpu_block;
                self.in_gpu_block = true;
                self.push_scope();
                for s in body { self.analyze_stmt(s, ret_ty); }
                self.pop_scope();
                self.in_gpu_block = was;
            }

            Stmt::Break | Stmt::Continue => {
                if self.loop_depth == 0 {
                    self.error(format!("`{}` outside of loop", if matches!(stmt, Stmt::Break) { "break" } else { "continue" }));
                }
            }

            Stmt::Import(_) => {}
        }
    }

    fn check_gpu_restrictions(&mut self, expr: &Expr) {
        let forbidden = ["file_read", "file_write", "file_open_read", "file_open_write",
                         "file_read_line", "file_write_line", "file_close", "load_model"];
        if let Expr::Call { callee, .. } = expr {
            if forbidden.contains(&callee.as_str()) {
                self.error(format!(
                    "GPU block: `{}` is not allowed inside gpu blocks", callee
                ));
            }
        }
    }

    pub fn infer_expr(&mut self, expr: &Expr) -> Type {
        match expr {
            Expr::IntLit(_)    => Type::Int64,
            Expr::FloatLit(_)  => Type::Float64,
            Expr::StringLit(_) => Type::Str,
            Expr::BoolLit(_)   => Type::Bool,

            Expr::Ident(name) => {
                match self.lookup_var(name).cloned() {
                    Some(info) => info.ty,
                    None => {
                        self.error(format!("Undefined variable `{}`", name));
                        Type::Unknown
                    }
                }
            }

            Expr::BinOp { op, lhs, rhs } => {
                let lt = self.infer_expr(lhs);
                let rt = self.infer_expr(rhs);
                match op {
                    BinOp::EqEq | BinOp::NotEq => Type::Bool,
                    BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => Type::Bool,
                    BinOp::And | BinOp::Or => Type::Bool,
                    BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor
                    | BinOp::Shl  | BinOp::Shr => Type::Int64,
                    _ => {
                        if lt != rt && lt != Type::Unknown && rt != Type::Unknown {
                            self.error(format!("Type mismatch: `{}` vs `{}`", lt, rt));
                        }
                        lt
                    }
                }
            }

            Expr::UnOp { op, expr } => {
                let ty = self.infer_expr(expr);
                match op {
                    UnOp::Not    => Type::Bool,
                    UnOp::Neg    => ty,
                    UnOp::BitNot => Type::Int64,
                }
            }

            Expr::Call { callee, args } => {
                let arg_types: Vec<Type> = args.iter().map(|a| self.infer_expr(a)).collect();
                if self.in_gpu_block {
                    for (i, at) in arg_types.iter().enumerate() {
                        if matches!(at, Type::Tensor(hw) if *hw.as_ref() == HwTarget::Cpu) {
                            self.error(format!(
                                "Arg {} to `{}`: `tensor@cpu` in gpu block — use `.toGpu()` first",
                                i + 1, callee
                            ));
                        }
                    }
                }
                if let Some((_, ret)) = self.functions.get(callee).cloned() {
                    // Pool polymorphism: a builtin that produces a tensor
                    // produces it in the pool its operands live in, so
                    // `mlx_add(gpu_a, gpu_b)` is `tensor@gpu`. Without this the
                    // result of every builtin was typed `tensor@cpu` and the
                    // next step in a gpu chain was rejected — which is why this
                    // bootstrap could not compile the pool-polymorphism
                    // reference example. Mirrors finalize_pool_poly() in
                    // src/semantic/checker.ionic.
                    //
                    // User functions are deliberately excluded: a
                    // `@cpu fn(tensor@cpu x)` is a promise about where the work
                    // happens, not a description of its operands.
                    if self.builtins.contains(callee) {
                        if matches!(ret, Type::Tensor(_)) {
                            if let Some(hw) = arg_types.iter().find_map(|t| match t {
                                Type::Tensor(hw) => Some(hw.clone()),
                                _ => None,
                            }) {
                                return Type::Tensor(hw);
                            }
                        }
                    }
                    ret
                } else {
                    // Unregistered names may be extern C symbols linked at
                    // runtime (getenv, str_slice, math, ML, …). Mirror the
                    // self-hosted checker's `fn_ret_ty`, which returns
                    // TY_UNKNOWN for these rather than guessing a type.
                    Type::Unknown
                }
            }

            Expr::MethodCall { obj, method, args } => {
                let obj_ty = self.infer_expr(obj);
                for a in args { self.infer_expr(a); }
                match (&obj_ty, method.as_str()) {
                    (Type::Tensor(_), "mean" | "sum") => Type::Float64,
                    (Type::Tensor(hw), "pow") => Type::Tensor(hw.clone()),
                    (Type::Array(_), "len") => Type::Int64,
                    (Type::Array(_), "push") => Type::Void,
                    (Type::Array(inner), "pop") => *inner.clone(),
                    (Type::Str, "len") => Type::Int64,
                    // model.forward(tensor) -> tensor (same hw target)
                    (Type::Model(hw), "forward") => Type::Tensor(hw.clone()),
                    // model.to_gpu() / model.to_cpu()
                    (Type::Model(_), "to_gpu") => Type::Model(Box::new(HwTarget::Gpu)),
                    (Type::Model(_), "to_cpu") => Type::Model(Box::new(HwTarget::Cpu)),
                    _ => Type::Unknown,
                }
            }

            Expr::StructLit { name, fields } => {
                if let Some(def_fields) = self.structs.get(name).cloned() {
                    for (fname, val) in fields {
                        let val_ty = self.infer_expr(val);
                        if let Some((_, expected)) = def_fields.iter().find(|(f, _)| f == fname) {
                            if val_ty != *expected && val_ty != Type::Unknown {
                                self.error(format!(
                                    "Struct `{}` field `{}`: expected `{}`, got `{}`",
                                    name, fname, expected, val_ty
                                ));
                            }
                        } else {
                            self.error(format!("No field `{}` on struct `{}`", fname, name));
                        }
                    }
                    Type::Struct(name.clone())
                } else {
                    self.error(format!("Unknown struct `{}`", name));
                    Type::Unknown
                }
            }

            Expr::FieldAccess { obj, field } => {
                let obj_ty = self.infer_expr(obj);
                if field == "len" {
                    match &obj_ty {
                        Type::Array(_) | Type::Str => return Type::Int64,
                        _ => {}
                    }
                }
                match &obj_ty {
                    Type::Struct(sname) => {
                        if let Some(fields) = self.structs.get(sname).cloned() {
                            if let Some((_, ty)) = fields.iter().find(|(f, _)| f == field) {
                                ty.clone()
                            } else {
                                self.error(format!("No field `{}` on struct `{}`", field, sname));
                                Type::Unknown
                            }
                        } else {
                            Type::Unknown
                        }
                    }
                    _ => Type::Unknown,
                }
            }

            Expr::Index { obj, idx } => {
                let obj_ty = self.infer_expr(obj);
                self.infer_expr(idx);
                match obj_ty {
                    Type::Array(inner) => *inner,
                    Type::Str => Type::Int64, // char as int
                    _ => Type::Unknown,
                }
            }

            Expr::ArrayLit(elems) => {
                if elems.is_empty() {
                    return Type::Array(Box::new(Type::Unknown));
                }
                let first = self.infer_expr(&elems[0]);
                for e in &elems[1..] {
                    let t = self.infer_expr(e);
                    if t != first && t != Type::Unknown {
                        self.error(format!("Array elements must have uniform type: `{}` vs `{}`", first, t));
                    }
                }
                Type::Array(Box::new(first))
            }

            Expr::ToGpu(inner) => {
                let ty = self.infer_expr(inner);
                match ty {
                    Type::Tensor(_) => Type::Tensor(Box::new(HwTarget::Gpu)),
                    other => { self.error(format!("`.toGpu()` requires a tensor, got `{}`", other)); Type::Unknown }
                }
            }

            Expr::ToCpu(inner) => {
                let ty = self.infer_expr(inner);
                match ty {
                    Type::Tensor(_) => Type::Tensor(Box::new(HwTarget::Cpu)),
                    other => { self.error(format!("`.toCpu()` requires a tensor, got `{}`", other)); Type::Unknown }
                }
            }
        }
    }
}
