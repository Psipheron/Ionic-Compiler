use std::collections::{HashMap, HashSet};
use crate::ast::*;

/// Struct layout: field names in order (all fields are i64-sized for simplicity in Phase 1)
#[derive(Debug, Clone)]
pub struct StructLayout {
    pub fields: Vec<(String, Type)>, // ordered field list
}

pub struct Codegen {
    out: String,
    reg: usize,
    slot_id: usize,                                 // unique alloca suffix
    str_lits: Vec<(String, usize)>,                 // (escaped, raw_byte_count)
    vars: Vec<HashMap<String, (String, Type)>>,     // name -> (alloca_ptr, type)
    fns: HashMap<String, (Vec<Type>, Type)>,
    /// Built-in signatures (currently the MLX surface). Shared with the checker
    /// via `semantic::mlx_builtins` so both phases agree on return types.
    builtins: HashMap<String, (Vec<Type>, Type)>,
    structs: HashMap<String, StructLayout>,
    current_fn_ret: Type,
    label: usize,
    // Break/continue target label stacks
    break_labels: Vec<String>,
    continue_labels: Vec<String>,
    // Top-level variable names declared as LLVM globals (accessible from functions)
    global_var_names: HashSet<String>,
}

impl Codegen {
    pub fn new() -> Self {
        Codegen {
            out: String::new(),
            reg: 0,
            slot_id: 0,
            str_lits: Vec::new(),
            vars: vec![HashMap::new()],
            fns: HashMap::new(),
            builtins: HashMap::new(),
            structs: HashMap::new(),
            current_fn_ret: Type::Void,
            label: 0,
            break_labels: Vec::new(),
            continue_labels: Vec::new(),
            global_var_names: HashSet::new(),
        }
    }

    fn fresh_slot(&mut self, name: &str) -> String {
        self.slot_id += 1;
        format!("%{}.s{}", name, self.slot_id)
    }

    // ── Output ────────────────────────────────────────────────────────────────

    fn emit(&mut self, line: &str) {
        self.out.push_str(line);
        self.out.push('\n');
    }

    fn fresh_reg(&mut self) -> String {
        self.reg += 1;
        format!("%r{}", self.reg)
    }

    fn fresh_label(&mut self) -> String {
        self.label += 1;
        format!("lbl{}", self.label)
    }

    fn intern_str(&mut self, s: &str) -> usize {
        let escaped = s
            .replace('\\', "\\5C")
            .replace('\n', "\\0A")
            .replace('\t', "\\09")
            .replace('"', "\\22");
        let bc = raw_byte_count(&escaped);
        self.str_lits.push((escaped, bc));
        self.str_lits.len() - 1
    }

    // ── Type system ───────────────────────────────────────────────────────────

    fn llvm_ty(ty: &Type) -> &'static str {
        match ty {
            Type::Int64 | Type::Bool => "i64",
            Type::Float64 => "double",
            Type::Str | Type::Array(_) | Type::Tensor(_) | Type::Model(_) | Type::Struct(_) => "ptr",
            Type::Void | Type::Unknown => "void",
        }
    }

    fn struct_field_offset(layout: &StructLayout, field: &str) -> Option<usize> {
        layout.fields.iter().position(|(f, _)| f == field)
    }

    fn struct_size(layout: &StructLayout) -> usize {
        layout.fields.len() // each field is 8 bytes (i64/ptr)
    }

    // ── Ptr ↔ i64 coercions for array element storage ────────────────────────

    fn is_ptr_ty(ty: &Type) -> bool {
        matches!(ty, Type::Str | Type::Array(_) | Type::Struct(_) | Type::Tensor(_) | Type::Model(_))
    }

    fn ptr_to_i64(&mut self, val: &str, ty: &Type) -> String {
        if Self::is_ptr_ty(ty) {
            let r = self.fresh_reg();
            self.emit(&format!("  {} = ptrtoint ptr {} to i64", r, val));
            r
        } else {
            val.to_string()
        }
    }

    fn i64_to_elem(&mut self, val: &str, ty: &Type) -> String {
        if Self::is_ptr_ty(ty) {
            let r = self.fresh_reg();
            self.emit(&format!("  {} = inttoptr i64 {} to ptr", r, val));
            r
        } else {
            val.to_string()
        }
    }

    // ── Global variable type inference ───────────────────────────────────────
    // Returns (llvm_ty_str, zero_init_str, ast_Type) for a top-level variable.
    fn infer_global_ty(init: &Expr) -> (&'static str, String, Type) {
        match init {
            Expr::IntLit(n)   => ("i64",  n.to_string(),    Type::Int64),
            Expr::BoolLit(b)  => ("i64",  if *b {"1"} else {"0"}.to_string(), Type::Int64),
            Expr::FloatLit(f) => ("double", format!("{:.17e}", f), Type::Float64),
            Expr::StringLit(_) => ("ptr", "null".to_string(), Type::Str),
            Expr::ArrayLit(elems) => {
                let inner = match elems.first() {
                    Some(Expr::IntLit(_)) | Some(Expr::BoolLit(_)) => Type::Int64,
                    Some(Expr::FloatLit(_)) => Type::Float64,
                    Some(Expr::StringLit(_)) => Type::Str,
                    _ => Type::Int64, // conservative default
                };
                ("ptr", "null".to_string(), Type::Array(Box::new(inner)))
            }
            _ => ("ptr", "null".to_string(), Type::Str),
        }
    }

    // ── Scope helpers ─────────────────────────────────────────────────────────

    fn push_scope(&mut self) { self.vars.push(HashMap::new()); }
    fn pop_scope(&mut self)  { self.vars.pop(); }

    fn declare_var(&mut self, name: &str, slot: String, ty: Type) {
        if let Some(scope) = self.vars.last_mut() {
            scope.insert(name.to_string(), (slot, ty));
        }
    }

    fn lookup_var(&self, name: &str) -> Option<(String, Type)> {
        for scope in self.vars.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v.clone());
            }
        }
        None
    }

    // ── Public entry ──────────────────────────────────────────────────────────

    pub fn generate(&mut self, program: &Program) -> String {
        // Register struct layouts
        for s in &program.structs {
            let fields: Vec<(String, Type)> = s.fields.iter()
                .map(|f| (f.name.clone(), f.ty.to_type()))
                .collect();
            self.structs.insert(s.name.clone(), StructLayout { fields });
        }

        crate::semantic::mlx_builtins(&mut self.builtins);

        // Register function signatures
        for f in &program.fns {
            let pt: Vec<Type> = f.params.iter().map(|p| p.ty.to_type()).collect();
            self.fns.insert(f.name.clone(), (pt, f.ret_ty.to_type()));
        }

        self.emit("; Ionic compiled output — LLVM 15+ opaque pointers");
        self.emit("; Generated by ionic v0.1.0");
        self.emit("");

        // Format string constants
        self.emit("@fmt_int   = private constant [6 x i8]  c\"%lld\\0A\\00\"");
        self.emit("@fmt_float = private constant [5 x i8]  c\"%lf\\0A\\00\"");
        self.emit("@fmt_str   = private constant [4 x i8]  c\"%s\\0A\\00\"");
        self.emit("@fmt_bool_t = private constant [6 x i8] c\"true\\0A\\00\"");
        self.emit("@fmt_bool_f = private constant [7 x i8] c\"false\\0A\\00\"");
        self.emit("");

        // External declarations
        self.emit("declare i32  @printf(ptr, ...)");
        self.emit("declare i32  @puts(ptr)");
        self.emit("declare ptr  @malloc(i64)");
        self.emit("declare void @free(ptr)");
        self.emit("declare ptr  @realloc(ptr, i64)");
        self.emit("declare ptr  @memcpy(ptr, ptr, i64)");
        self.emit("declare i64  @strlen(ptr)");
        self.emit("declare ptr  @strcat(ptr, ptr)");
        self.emit("declare ptr  @strcpy(ptr, ptr)");
        self.emit("declare i32  @strcmp(ptr, ptr)");
        self.emit("declare ptr  @strdup(ptr)");
        self.emit("declare i32  @sprintf(ptr, ptr, ...)");
        self.emit("declare i32  @snprintf(ptr, i64, ptr, ...)"); // size_t ABI issues; use sprintf where possible
        self.emit("declare double @sqrt(double)");
        self.emit("declare double @fabs(double)");
        self.emit("declare void @exit(i32)");
        // File I/O
        self.emit("declare ptr  @fopen(ptr, ptr)");
        self.emit("declare i32  @fclose(ptr)");
        self.emit("declare ptr  @fgets(ptr, i32, ptr)");
        self.emit("declare i32  @fputs(ptr, ptr)");
        self.emit("declare i32  @feof(ptr)");
        self.emit("declare i32  @fseek(ptr, i64, i32)");
        self.emit("declare i64  @ftell(ptr)");
        self.emit("declare i64  @fread(ptr, i64, i64, ptr)");
        // ML model runtime (ionic_model_runtime.c)
        self.emit("declare ptr  @ionic_load_model(ptr)");
        self.emit("declare ptr  @ionic_model_forward(ptr, ptr)");
        self.emit("declare void @ionic_model_free(ptr)");
        self.emit("declare ptr  @ionic_piper_forward(ptr, ptr, double, double, double)");
        self.emit("declare void @ionic_write_wav(ptr, ptr, i64, i64)");
        self.emit("declare ptr  @ionic_gguf_generate(ptr, ptr, i64)");
        self.emit("declare void @ionic_gguf_set_temp(ptr, double)");
        self.emit("declare void @ionic_gguf_set_top_p(ptr, double)");
        self.emit("declare ptr  @ionic_fgets_stdin(ptr, i32)");
        // MLX backend declarations, derived from the shared signature table so
        // a builtin added there is declared here automatically.
        let mut mlx_decls: Vec<String> = Vec::new();
        let mut mlx_names: Vec<String> = self.builtins.keys().cloned().collect();
        mlx_names.sort();
        for name in &mlx_names {
            let (params, ret) = &self.builtins[name];
            // `mlx_last_error` returns ptr, and a float64 return is actually an
            // i64 bit pattern that the caller bitcasts back to double.
            let lret = match ret {
                Type::Void => "void",
                Type::Str => "ptr",
                _ => "i64",
            };
            let plist: Vec<&str> = params.iter().map(|p| match p {
                // float64 crosses the FFI boundary as an IEEE-754 bit pattern in
                // a GP register (see the same note in ionic_model_runtime.c), so
                // the declaration must say i64, not double. Declaring `double`
                // would place the value in a D register while the C side reads
                // X — the call succeeds and silently returns garbage.
                Type::Float64 => "i64",
                Type::Str | Type::Array(_) | Type::Tensor(_) | Type::Model(_) => "ptr",
                _ => "i64",
            }).collect();
            mlx_decls.push(format!("declare {} @ionic_{}({})", lret, name, plist.join(", ")));
        }
        for d in &mlx_decls {
            self.emit(d);
        }
        self.emit("declare i64  @ionic_file_write_binary(ptr, ptr)");
        self.emit("declare i64  @ionic_arr_reset(ptr)");
        self.emit("declare i64  @str_to_float64_bits(ptr)");  // float literal parser helper
        // IEEE-754 bit-pattern float helpers (used by the self-hosted optimizer
        // for compile-time float folding); defined unprefixed in the runtime C.
        self.emit("declare i64  @fadd_bits(i64, i64)");
        self.emit("declare i64  @fsub_bits(i64, i64)");
        self.emit("declare i64  @fmul_bits(i64, i64)");
        self.emit("declare i64  @fdiv_bits(i64, i64)");
        self.emit("declare i64  @sqrt_bits(i64)");
        self.emit("declare i64  @fabs_bits(i64)");
        self.emit("declare i64  @floor_bits(i64)");
        self.emit("declare i64  @ceil_bits(i64)");
        self.emit("declare i64  @pow_bits(i64, i64)");
        self.emit("declare i64  @int_to_fbits(i64)");
        // System runtime
        self.emit("declare void @ionic_runtime_init(i32, ptr)");
        self.emit("declare ptr  @ionic_get_arg(i64)");
        self.emit("declare i64  @ionic_cpu_core_count()");
        self.emit("declare i32  @access(ptr, i32)");  // POSIX file_exists
        self.emit("declare i64  @ionic_system(ptr)");
        // String / system builtins used by the self-hosted compiler sources.
        // These map plain Ionic names onto their ionic_* runtime symbols.
        self.emit("declare ptr  @ionic_getenv(ptr)");
        self.emit("declare ptr  @ionic_read_line()");
        self.emit("declare ptr  @ionic_str_slice(ptr, i64, i64)");
        self.emit("declare ptr  @ionic_str_replace(ptr, ptr, ptr)");
        self.emit("declare i64  @ionic_str_contains(ptr, ptr)");
        self.emit("declare i64  @ionic_str_starts_with(ptr, ptr)");
        self.emit("declare i64  @ionic_str_ends_with(ptr, ptr)");
        self.emit("declare ptr  @ionic_format(ptr)");
        self.emit("declare i64  @ionic_str_to_int64(ptr)");
        self.emit("declare i64  @ionic_str_hash(ptr)");
        self.emit("declare i64  @ionic_target_is_linux()");
        self.emit("declare double @ionic_pow(double, double)");
        self.emit("declare double @ionic_floor(double)");
        self.emit("declare double @ionic_ceil(double)");
        self.emit("");

        // Emit Ionic runtime helpers
        self.emit_runtime_helpers();

        // Emit struct type aliases (opaque, all fields are i64)
        let struct_names: Vec<(String, usize)> = self.structs
            .iter()
            .map(|(n, l)| (n.clone(), Self::struct_size(l)))
            .collect();
        for (name, size) in &struct_names {
            // Each field is 8 bytes
            self.emit(&format!(
                "; struct {} : {} fields ({} bytes)",
                name, size, size * 8
            ));
        }
        if !struct_names.is_empty() { self.emit(""); }

        // Pre-emit ALL top-level let/mut variables as LLVM globals so that function
        // bodies can read and write them.  Scalars get their compile-time value;
        // everything else is null-initialized and runtime-assigned in main().
        let mut had_globals = false;
        for stmt in &program.top_level {
            if let Stmt::Let { mutable, name, ty: _, init, hw: _ } = stmt {
                let (llty, zero_init, var_ty) = Self::infer_global_ty(init);
                let vis = if !mutable && llty == "i64" { "private global" } else { "global" };
                self.emit(&format!("@{} = {} {} {}, align 8", name, vis, llty, zero_init));
                self.declare_var(name, format!("@{}", name), var_ty);
                self.global_var_names.insert(name.clone());
                had_globals = true;
            }
        }
        if had_globals { self.emit(""); }

        let has_user_main = program.fns.iter()
            .any(|f| f.name == "main" && f.params.is_empty());

        let fns: Vec<FnDef> = program.fns.clone();
        for f in &fns {
            self.emit_fn(f);
        }

        let top: Vec<Stmt> = program.top_level.clone();
        self.emit_main(&top, has_user_main);

        // Flush interned string literals
        let lits = self.str_lits.clone();
        for (i, (escaped, bc)) in lits.iter().enumerate() {
            self.emit(&format!(
                "@str{} = private constant [{} x i8] c\"{}\\00\"",
                i, bc + 1, escaped
            ));
        }

        self.out.clone()
    }

    fn emit_runtime_helpers(&mut self) {
        // Array layout: 24-byte header [len:i64][cap:i64][data_ptr:ptr] + separate data heap.
        // Keeping the header address stable lets push realloc data without invalidating callers.
        self.emit("; --- Ionic runtime: dynamic array ---");
        self.emit("define ptr @ionic_array_new(i64 %elem_size, i64 %cap) {");
        self.emit("entry:");
        self.emit("  %hdr = call ptr @malloc(i64 24)");       // 3 x i64 header
        self.emit("  %data_size = mul i64 %elem_size, %cap");
        self.emit("  %data = call ptr @malloc(i64 %data_size)");
        self.emit("  store i64 0, ptr %hdr, align 8");                        // len = 0
        self.emit("  %cap_f = getelementptr i64, ptr %hdr, i64 1");
        self.emit("  store i64 %cap, ptr %cap_f, align 8");                   // cap
        self.emit("  %dat_f = getelementptr i64, ptr %hdr, i64 2");
        self.emit("  store ptr %data, ptr %dat_f, align 8");                  // data_ptr
        self.emit("  ret ptr %hdr");
        self.emit("}");
        self.emit("");

        self.emit("define i64 @ionic_array_len(ptr %arr) {");
        self.emit("entry:");
        self.emit("  %len = load i64, ptr %arr, align 8");
        self.emit("  ret i64 %len");
        self.emit("}");
        self.emit("");

        self.emit("define i64 @ionic_array_get(ptr %arr, i64 %idx) {");
        self.emit("entry:");
        self.emit("  %dat_f = getelementptr i64, ptr %arr, i64 2");
        self.emit("  %dp = load ptr, ptr %dat_f, align 8");
        self.emit("  %ep = getelementptr i64, ptr %dp, i64 %idx");
        self.emit("  %val = load i64, ptr %ep, align 8");
        self.emit("  ret i64 %val");
        self.emit("}");
        self.emit("");

        self.emit("define void @ionic_array_set(ptr %arr, i64 %idx, i64 %val) {");
        self.emit("entry:");
        self.emit("  %dat_f = getelementptr i64, ptr %arr, i64 2");
        self.emit("  %dp = load ptr, ptr %dat_f, align 8");
        self.emit("  %ep = getelementptr i64, ptr %dp, i64 %idx");
        self.emit("  store i64 %val, ptr %ep, align 8");
        self.emit("  ret void");
        self.emit("}");
        self.emit("");

        // push: doubles capacity via realloc when full; header address is stable.
        self.emit("define void @ionic_array_push(ptr %arr, i64 %val) {");
        self.emit("entry:");
        self.emit("  %len     = load i64, ptr %arr, align 8");
        self.emit("  %cap_f   = getelementptr i64, ptr %arr, i64 1");
        self.emit("  %cap     = load i64, ptr %cap_f, align 8");
        self.emit("  %dat_f   = getelementptr i64, ptr %arr, i64 2");
        self.emit("  %dp      = load ptr, ptr %dat_f, align 8");
        self.emit("  %full    = icmp sge i64 %len, %cap");
        self.emit("  br i1 %full, label %grow, label %store");
        self.emit("grow:");
        self.emit("  %new_cap  = mul i64 %cap, 2");
        self.emit("  %new_sz   = mul i64 %new_cap, 8");
        self.emit("  %new_dp   = call ptr @realloc(ptr %dp, i64 %new_sz)");
        self.emit("  store i64 %new_cap, ptr %cap_f, align 8");
        self.emit("  store ptr %new_dp, ptr %dat_f, align 8");
        self.emit("  br label %store");
        self.emit("store:");
        self.emit("  %cur_dp  = load ptr, ptr %dat_f, align 8");
        self.emit("  %ep      = getelementptr i64, ptr %cur_dp, i64 %len");
        self.emit("  store i64 %val, ptr %ep, align 8");
        self.emit("  %nlen    = add i64 %len, 1");
        self.emit("  store i64 %nlen, ptr %arr, align 8");
        self.emit("  ret void");
        self.emit("}");
        self.emit("");

        // ail_str_concat(a, b) -> ptr
        self.emit("define ptr @ionic_str_concat(ptr %a, ptr %b) {");
        self.emit("entry:");
        self.emit("  %la = call i64 @strlen(ptr %a)");
        self.emit("  %lb = call i64 @strlen(ptr %b)");
        self.emit("  %tot1 = add i64 %la, %lb");
        self.emit("  %tot = add i64 %tot1, 1");
        self.emit("  %buf = call ptr @malloc(i64 %tot)");
        self.emit("  call ptr @strcpy(ptr %buf, ptr %a)");
        self.emit("  call ptr @strcat(ptr %buf, ptr %b)");
        self.emit("  ret ptr %buf");
        self.emit("}");
        self.emit("");

        // ail_int64_to_str(n) -> ptr
        self.emit("@fmt_lld = private constant [5 x i8] c\"%lld\\00\"");
        self.emit("define ptr @ionic_int64_to_str(i64 %n) {");
        self.emit("entry:");
        self.emit("  %buf = call ptr @malloc(i64 32)");
        self.emit("  call i32 (ptr, ptr, ...) @sprintf(ptr %buf, ptr @fmt_lld, i64 %n)");
        self.emit("  ret ptr %buf");
        self.emit("}");
        self.emit("");

        // ail_float64_to_str(f) -> ptr
        self.emit("@fmt_lf = private constant [4 x i8] c\"%lf\\00\"");
        self.emit("define ptr @ionic_float64_to_str(double %f) {");
        self.emit("entry:");
        self.emit("  %buf = call ptr @malloc(i64 64)");
        self.emit("  call i32 (ptr, ptr, ...) @sprintf(ptr %buf, ptr @fmt_lf, double %f)");
        self.emit("  ret ptr %buf");
        self.emit("}");
        self.emit("");

        // ail_char_to_str(c) -> ptr
        self.emit("define ptr @ionic_char_to_str(i64 %c) {");
        self.emit("entry:");
        self.emit("  %buf = call ptr @malloc(i64 2)");
        self.emit("  %c8 = trunc i64 %c to i8");
        self.emit("  store i8 %c8, ptr %buf, align 1");
        self.emit("  %nul_ptr = getelementptr i8, ptr %buf, i64 1");
        self.emit("  store i8 0, ptr %nul_ptr, align 1");
        self.emit("  ret ptr %buf");
        self.emit("}");
        self.emit("");

        // File I/O wrappers
        self.emit("@mode_r = private constant [2 x i8] c\"r\\00\"");
        self.emit("@mode_w = private constant [2 x i8] c\"w\\00\"");
        self.emit("@empty_str = private constant [1 x i8] c\"\\00\"");

        // file_read(path) -> string (reads entire file via fseek/ftell/fread).
        // Must guard against fopen returning NULL: a missing file previously
        // segfaulted in flockfile() via fseek(NULL). Mirror the C runtime's
        // `if (!f) return "";` behaviour by returning a static empty string.
        self.emit("define ptr @ionic_file_read(ptr %path) {");
        self.emit("entry:");
        self.emit("  %fp = call ptr @fopen(ptr %path, ptr @mode_r)");
        self.emit("  %missing = icmp eq ptr %fp, null");
        self.emit("  br i1 %missing, label %no_file, label %read_file");
        self.emit("no_file:");
        self.emit("  ret ptr @empty_str");
        self.emit("read_file:");
        self.emit("  call i32 @fseek(ptr %fp, i64 0, i32 2)");   // SEEK_END=2
        self.emit("  %fsz = call i64 @ftell(ptr %fp)");
        self.emit("  call i32 @fseek(ptr %fp, i64 0, i32 0)");   // SEEK_SET=0
        self.emit("  %bufsz = add i64 %fsz, 1");
        self.emit("  %buf = call ptr @malloc(i64 %bufsz)");
        self.emit("  call i64 @fread(ptr %buf, i64 1, i64 %fsz, ptr %fp)");
        self.emit("  %end = getelementptr i8, ptr %buf, i64 %fsz");
        self.emit("  store i8 0, ptr %end, align 1");
        self.emit("  call i32 @fclose(ptr %fp)");
        self.emit("  ret ptr %buf");
        self.emit("}");
        self.emit("");
    }

    fn emit_fn(&mut self, f: &FnDef) {
        self.reg = 0;
        self.label = 0;
        self.slot_id = 0;
        self.push_scope();
        self.current_fn_ret = f.ret_ty.to_type();

        let params_ir: Vec<String> = f.params.iter().map(|p| {
            let llty = Self::llvm_ty(&p.ty.to_type());
            // Bool params come in as i64 at the boundary for simplicity
            format!("{} %{}", llty, p.name)
        }).collect();

        let ret_llvm = Self::llvm_ty(&f.ret_ty.to_type());

        // Rename user's `fn main()` to avoid collision with the runtime wrapper @main
        let emit_name = if f.name == "main" && f.params.is_empty() {
            "ionic_user_main".to_string()
        } else {
            f.name.clone()
        };

        // Emit hw placement metadata as IR comment on the function
        if let Some(hw) = &f.hw {
            self.emit(&format!("  {}", hw.ir_comment()));
        }
        self.emit(&format!("define {} @{}({}) {{", ret_llvm, emit_name, params_ir.join(", ")));
        self.emit("entry:");

        for p in &f.params {
            let ty = p.ty.to_type();
            let llty = Self::llvm_ty(&ty);
            let slot = self.fresh_slot(&p.name);
            self.emit(&format!("  {} = alloca {}, align 8", slot, llty));
            self.emit(&format!("  store {} %{}, ptr {}, align 8", llty, p.name, slot));
            self.declare_var(&p.name, slot, ty);
        }

        for stmt in &f.body.clone() {
            self.emit_stmt(stmt);
        }

        if f.ret_ty.to_type() == Type::Void {
            self.emit("  ret void");
        }
        self.emit("}");
        self.emit("");
        self.pop_scope();
    }

    fn emit_main(&mut self, stmts: &[Stmt], has_user_main: bool) {
        self.reg = 0;
        self.label = 0;
        self.slot_id = 0;
        self.current_fn_ret = Type::Void;
        self.emit("define i32 @main(i32 %_argc, ptr %_argv) {");
        self.emit("entry:");
        self.emit("  call void @ionic_runtime_init(i32 %_argc, ptr %_argv)");
        for stmt in stmts {
            self.emit_stmt(stmt);
        }
        if has_user_main {
            self.emit("  %_exit = call i64 @ionic_user_main()");
            self.emit("  %_exit32 = trunc i64 %_exit to i32");
            self.emit("  ret i32 %_exit32");
        } else {
            self.emit("  ret i32 0");
        }
        self.emit("}");
        self.emit("");
    }

    // ── Statement codegen ─────────────────────────────────────────────────────

    fn emit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let { mutable: _, name, ty, init, hw: _ } => {
                let (val, vty) = self.emit_expr(init);
                let resolved = ty.as_ref().map(|t| t.to_type()).unwrap_or_else(|| vty.clone());
                let llty = Self::llvm_ty(&resolved);
                if self.global_var_names.contains(name.as_str()) && self.vars.len() == 1 {
                    // Top-level (main scope) global: store init value into the LLVM global.
                    // Only applies at the outermost scope (vars.len()==1); function-local
                    // variables with the same name get their own alloca and shadow the global.
                    self.emit(&format!("  store {} {}, ptr @{}, align 8", llty, val, name));
                    // Refresh scope entry with the resolved (more precise) type.
                    self.declare_var(name, format!("@{}", name), resolved);
                } else {
                    let slot = self.fresh_slot(name);
                    self.emit(&format!("  {} = alloca {}, align 8", slot, llty));
                    self.emit(&format!("  store {} {}, ptr {}, align 8", llty, val, slot));
                    self.declare_var(name, slot, resolved);
                }
            }

            Stmt::Assign { target, value } => {
                let (val, vty) = self.emit_expr(value);
                let llty = Self::llvm_ty(&vty);
                match target {
                    AssignTarget::Ident(name) => {
                        if let Some((slot, _)) = self.lookup_var(name) {
                            self.emit(&format!("  store {} {}, ptr {}, align 8", llty, val, slot));
                        }
                    }
                    AssignTarget::Field(obj, field) => {
                        let (obj_val, obj_ty) = self.emit_expr(obj);
                        if let Type::Struct(sname) = &obj_ty {
                            if let Some(layout) = self.structs.get(sname).cloned() {
                                if let Some(idx) = Self::struct_field_offset(&layout, field) {
                                    let gep = self.fresh_reg();
                                    self.emit(&format!(
                                        "  {} = getelementptr i64, ptr {}, i64 {}",
                                        gep, obj_val, idx
                                    ));
                                    self.emit(&format!("  store {} {}, ptr {}, align 8", llty, val, gep));
                                }
                            }
                        }
                    }
                    AssignTarget::Index(obj, idx) => {
                        let (obj_val, _) = self.emit_expr(obj);
                        let (idx_val, _) = self.emit_expr(idx);
                        let i64_val = self.ptr_to_i64(&val, &vty);
                        self.emit(&format!(
                            "  call void @ionic_array_set(ptr {}, i64 {}, i64 {})",
                            obj_val, idx_val, i64_val
                        ));
                    }
                }
            }

            Stmt::ExprStmt(e) => { self.emit_expr(e); }

            Stmt::Return(expr) => {
                if let Some(e) = expr {
                    let (v, ty) = self.emit_expr(e);
                    self.emit(&format!("  ret {} {}", Self::llvm_ty(&ty), v));
                } else {
                    self.emit("  ret void");
                }
            }

            Stmt::If { cond, then_block, else_block } => {
                let (cond_v, cond_ty) = self.emit_expr(cond);
                // Condition might be i64 (bool as i64) — truncate to i1
                let cond_i1 = self.coerce_to_i1(&cond_v, &cond_ty);

                let then_lbl  = self.fresh_label();
                let merge_lbl = self.fresh_label();

                if else_block.is_some() {
                    let else_lbl = self.fresh_label();
                    self.emit(&format!("  br i1 {}, label %{}, label %{}", cond_i1, then_lbl, else_lbl));
                    self.emit(&format!("{}:", then_lbl));
                    self.push_scope();
                    for s in then_block { self.emit_stmt(s); }
                    self.pop_scope();
                    self.emit(&format!("  br label %{}", merge_lbl));
                    self.emit(&format!("{}:", else_lbl));
                    self.push_scope();
                    for s in else_block.as_ref().unwrap() { self.emit_stmt(s); }
                    self.pop_scope();
                    self.emit(&format!("  br label %{}", merge_lbl));
                } else {
                    self.emit(&format!("  br i1 {}, label %{}, label %{}", cond_i1, then_lbl, merge_lbl));
                    self.emit(&format!("{}:", then_lbl));
                    self.push_scope();
                    for s in then_block { self.emit_stmt(s); }
                    self.pop_scope();
                    self.emit(&format!("  br label %{}", merge_lbl));
                }
                self.emit(&format!("{}:", merge_lbl));
            }

            Stmt::While { cond, body } => {
                let cond_lbl = self.fresh_label();
                let body_lbl = self.fresh_label();
                let exit_lbl = self.fresh_label();

                self.break_labels.push(exit_lbl.clone());
                self.continue_labels.push(cond_lbl.clone());

                self.emit(&format!("  br label %{}", cond_lbl));
                self.emit(&format!("{}:", cond_lbl));
                let (cv, cty) = self.emit_expr(cond);
                let ci1 = self.coerce_to_i1(&cv, &cty);
                self.emit(&format!("  br i1 {}, label %{}, label %{}", ci1, body_lbl, exit_lbl));

                self.emit(&format!("{}:", body_lbl));
                self.push_scope();
                for s in body { self.emit_stmt(s); }
                self.pop_scope();
                self.emit(&format!("  br label %{}", cond_lbl));

                self.emit(&format!("{}:", exit_lbl));
                self.break_labels.pop();
                self.continue_labels.pop();
            }

            Stmt::For { var, start, end, body } => {
                let (sv, _) = self.emit_expr(start);
                let (ev, _) = self.emit_expr(end);

                let slot = self.fresh_slot(var);
                self.emit(&format!("  {} = alloca i64, align 8", slot));
                self.emit(&format!("  store i64 {}, ptr {}, align 8", sv, slot));

                let cond_lbl = self.fresh_label();
                let body_lbl = self.fresh_label();
                let incr_lbl = self.fresh_label();
                let exit_lbl = self.fresh_label();

                self.break_labels.push(exit_lbl.clone());
                self.continue_labels.push(incr_lbl.clone());

                self.emit(&format!("  br label %{}", cond_lbl));
                self.emit(&format!("{}:", cond_lbl));
                let cur = self.fresh_reg();
                self.emit(&format!("  {} = load i64, ptr {}, align 8", cur, slot));
                let cmp_i1 = self.fresh_reg();
                let cmp    = self.fresh_reg();
                self.emit(&format!("  {} = icmp slt i64 {}, {}", cmp_i1, cur, ev));
                self.emit(&format!("  {} = zext i1 {} to i64", cmp, cmp_i1));
                let br = self.fresh_reg();
                self.emit(&format!("  {} = trunc i64 {} to i1", br, cmp));
                self.emit(&format!("  br i1 {}, label %{}, label %{}", br, body_lbl, exit_lbl));

                self.emit(&format!("{}:", body_lbl));
                self.push_scope();
                self.declare_var(var, slot.clone(), Type::Int64);
                for s in body { self.emit_stmt(s); }
                self.pop_scope();
                self.emit(&format!("  br label %{}", incr_lbl));

                self.emit(&format!("{}:", incr_lbl));
                let cur2 = self.fresh_reg();
                let inc  = self.fresh_reg();
                self.emit(&format!("  {} = load i64, ptr {}, align 8", cur2, slot));
                self.emit(&format!("  {} = add i64 {}, 1", inc, cur2));
                self.emit(&format!("  store i64 {}, ptr {}, align 8", inc, slot));
                self.emit(&format!("  br label %{}", cond_lbl));

                self.emit(&format!("{}:", exit_lbl));
                self.break_labels.pop();
                self.continue_labels.pop();
            }

            Stmt::GpuBlock(body) => {
                self.emit("  ; === gpu block begin ===");
                self.push_scope();
                for s in body { self.emit_stmt(s); }
                self.pop_scope();
                self.emit("  ; === gpu block end ===");
            }

            Stmt::Break => {
                if let Some(lbl) = self.break_labels.last().cloned() {
                    self.emit(&format!("  br label %{}", lbl));
                    // Emit a dead block so LLVM doesn't complain about missing terminator
                    let dead = self.fresh_label();
                    self.emit(&format!("{}:", dead));
                }
            }

            Stmt::Continue => {
                if let Some(lbl) = self.continue_labels.last().cloned() {
                    self.emit(&format!("  br label %{}", lbl));
                    let dead = self.fresh_label();
                    self.emit(&format!("{}:", dead));
                }
            }

            Stmt::Import(_) => {}
        }
    }

    // ── Condition coercion ─────────────────────────────────────────────────────

    fn coerce_to_i1(&mut self, val: &str, ty: &Type) -> String {
        // All bools and ints are uniformly i64; truncate to i1 for branching
        let r = self.fresh_reg();
        self.emit(&format!("  {} = trunc i64 {} to i1", r, val));
        r
    }

    // ── Expression codegen ─────────────────────────────────────────────────────

    fn emit_expr(&mut self, expr: &Expr) -> (String, Type) {
        match expr {
            Expr::IntLit(n)    => (n.to_string(), Type::Int64),
            Expr::FloatLit(f)  => (format!("{:.17}", f), Type::Float64),
            Expr::BoolLit(b)   => (if *b { "1" } else { "0" }.to_string(), Type::Int64),

            Expr::StringLit(s) => {
                let idx = self.intern_str(s);
                let bc  = self.str_lits[idx].1;
                let reg = self.fresh_reg();
                self.emit(&format!(
                    "  {} = getelementptr [{} x i8], ptr @str{}, i32 0, i32 0",
                    reg, bc + 1, idx
                ));
                (reg, Type::Str)
            }

            Expr::Ident(name) => {
                if let Some((slot, ty)) = self.lookup_var(name) {
                    let llty = Self::llvm_ty(&ty);
                    if llty == "void" {
                        return ("0".to_string(), Type::Unknown);
                    }
                    let reg = self.fresh_reg();
                    self.emit(&format!("  {} = load {}, ptr {}, align 8", reg, llty, slot));
                    (reg, ty)
                } else {
                    (format!("0 ; undef:{}", name), Type::Unknown)
                }
            }

            Expr::BinOp { op, lhs, rhs } => {
                let (lv, lt) = self.emit_expr(lhs);
                let (rv, _rt) = self.emit_expr(rhs);
                let reg = self.fresh_reg();
                let is_float = lt == Type::Float64;
                let llty = Self::llvm_ty(&lt);

                let (instr, out_ty) = match op {
                    BinOp::Add   => (if is_float { format!("fadd {} {}, {}", llty, lv, rv) } else { format!("add {} {}, {}", llty, lv, rv) }, lt.clone()),
                    BinOp::Sub   => (if is_float { format!("fsub {} {}, {}", llty, lv, rv) } else { format!("sub {} {}, {}", llty, lv, rv) }, lt.clone()),
                    BinOp::Mul   => (if is_float { format!("fmul {} {}, {}", llty, lv, rv) } else { format!("mul {} {}, {}", llty, lv, rv) }, lt.clone()),
                    BinOp::Div   => (if is_float { format!("fdiv {} {}, {}", llty, lv, rv) } else { format!("sdiv {} {}, {}", llty, lv, rv) }, lt.clone()),
                    BinOp::Mod   => (if is_float { format!("frem {} {}, {}", llty, lv, rv) } else { format!("srem {} {}, {}", llty, lv, rv) }, lt.clone()),
                    // Comparisons: emit icmp (returns i1), then zext to i64
                    BinOp::EqEq  | BinOp::NotEq |
                    BinOp::Lt    | BinOp::Gt    |
                    BinOp::LtEq  | BinOp::GtEq  => {
                        let cmp_instr = match op {
                            BinOp::EqEq  => if is_float { format!("fcmp oeq {} {}, {}", llty, lv, rv) } else { format!("icmp eq {} {}, {}", llty, lv, rv) },
                            BinOp::NotEq => if is_float { format!("fcmp une {} {}, {}", llty, lv, rv) } else { format!("icmp ne {} {}, {}", llty, lv, rv) },
                            BinOp::Lt    => if is_float { format!("fcmp olt {} {}, {}", llty, lv, rv) } else { format!("icmp slt {} {}, {}", llty, lv, rv) },
                            BinOp::Gt    => if is_float { format!("fcmp ogt {} {}, {}", llty, lv, rv) } else { format!("icmp sgt {} {}, {}", llty, lv, rv) },
                            BinOp::LtEq  => if is_float { format!("fcmp ole {} {}, {}", llty, lv, rv) } else { format!("icmp sle {} {}, {}", llty, lv, rv) },
                            BinOp::GtEq  => if is_float { format!("fcmp oge {} {}, {}", llty, lv, rv) } else { format!("icmp sge {} {}, {}", llty, lv, rv) },
                            _ => unreachable!(),
                        };
                        // i1 result -> zext to i64 so everything stays uniformly i64
                        let i1_reg = self.fresh_reg();
                        self.emit(&format!("  {} = {}", i1_reg, cmp_instr));
                        self.emit(&format!("  {} = zext i1 {} to i64", reg, i1_reg));
                        return (reg, Type::Int64);
                    }
                    BinOp::And    => (format!("and i64 {}, {}", lv, rv), Type::Bool),
                    BinOp::Or     => (format!("or i64 {}, {}", lv, rv), Type::Bool),
                    BinOp::BitAnd => (format!("and i64 {}, {}", lv, rv), Type::Int64),
                    BinOp::BitOr  => (format!("or i64 {}, {}", lv, rv),  Type::Int64),
                    BinOp::BitXor => (format!("xor i64 {}, {}", lv, rv), Type::Int64),
                    BinOp::Shl    => (format!("shl i64 {}, {}", lv, rv),  Type::Int64),
                    BinOp::Shr    => (format!("ashr i64 {}, {}", lv, rv), Type::Int64),
                    _ => unreachable!(),
                };
                self.emit(&format!("  {} = {}", reg, instr));
                (reg, out_ty)
            }

            Expr::UnOp { op, expr } => {
                let (v, ty) = self.emit_expr(expr);
                let reg = self.fresh_reg();
                match op {
                    UnOp::Neg => {
                        if ty == Type::Float64 {
                            self.emit(&format!("  {} = fneg double {}", reg, v));
                        } else {
                            self.emit(&format!("  {} = sub i64 0, {}", reg, v));
                        }
                        (reg, ty)
                    }
                    UnOp::Not => {
                        self.emit(&format!("  {} = xor i64 {}, 1", reg, v));
                        (reg, Type::Bool)
                    }
                    UnOp::BitNot => {
                        self.emit(&format!("  {} = xor i64 {}, -1", reg, v));
                        (reg, Type::Int64)
                    }
                }
            }

            Expr::Call { callee, args } => {
                let arg_vals: Vec<(String, Type)> = args.iter().map(|a| self.emit_expr(a)).collect();
                self.emit_call(callee, &arg_vals)
            }

            Expr::MethodCall { obj, method, args } => {
                let (obj_val, obj_ty) = self.emit_expr(obj);
                let arg_vals: Vec<(String, Type)> = args.iter().map(|a| self.emit_expr(a)).collect();
                match (&obj_ty, method.as_str()) {
                    (Type::Array(_), "len") => {
                        let r = self.fresh_reg();
                        self.emit(&format!("  {} = call i64 @ionic_array_len(ptr {})", r, obj_val));
                        (r, Type::Int64)
                    }
                    (Type::Array(_), "push") => {
                        let (v, vt) = arg_vals[0].clone();
                        let i64_v = self.ptr_to_i64(&v, &vt);
                        self.emit(&format!("  call void @ionic_array_push(ptr {}, i64 {})", obj_val, i64_v));
                        ("0".to_string(), Type::Void)
                    }
                    (Type::Str, "len") => {
                        let r = self.fresh_reg();
                        self.emit(&format!("  {} = call i64 @strlen(ptr {})", r, obj_val));
                        (r, Type::Int64)
                    }
                    (Type::Model(hw), "forward") => {
                        let (tensor_v, _) = arg_vals.into_iter().next()
                            .unwrap_or_else(|| ("null".to_string(), Type::Unknown));
                        let r = self.fresh_reg();
                        self.emit(&format!(
                            "  {} = call ptr @ionic_model_forward(ptr {}, ptr {})",
                            r, obj_val, tensor_v
                        ));
                        (r, Type::Tensor(hw.clone()))
                    }
                    (Type::Model(_), "to_gpu") => {
                        self.emit("  ; model.to_gpu() — stub: Phase 1 returns same handle");
                        (obj_val, Type::Model(Box::new(HwTarget::Gpu)))
                    }
                    (Type::Model(_), "to_cpu") => {
                        self.emit("  ; model.to_cpu() — stub");
                        (obj_val, Type::Model(Box::new(HwTarget::Cpu)))
                    }
                    _ => {
                        // Generic dispatch
                        let r = self.fresh_reg();
                        self.emit(&format!("  {} = add i64 0, 0 ; stub method .{}", r, method));
                        (r, Type::Int64)
                    }
                }
            }

            Expr::StructLit { name, fields } => {
                let layout = self.structs.get(name).cloned();
                if let Some(layout) = layout {
                    let size = Self::struct_size(&layout);
                    let byte_size = (size * 8) as i64;
                    let ptr = self.fresh_reg();
                    self.emit(&format!("  {} = call ptr @malloc(i64 {})", ptr, byte_size));
                    // Store each field
                    for (fname, val_expr) in fields {
                        let (val, vty) = self.emit_expr(val_expr);
                        if let Some(idx) = Self::struct_field_offset(&layout, fname) {
                            let gep = self.fresh_reg();
                            let llty = Self::llvm_ty(&vty);
                            self.emit(&format!(
                                "  {} = getelementptr i64, ptr {}, i64 {}",
                                gep, ptr, idx
                            ));
                            self.emit(&format!("  store {} {}, ptr {}, align 8", llty, val, gep));
                        }
                    }
                    (ptr, Type::Struct(name.clone()))
                } else {
                    ("null".to_string(), Type::Unknown)
                }
            }

            Expr::FieldAccess { obj, field } => {
                let (obj_val, obj_ty) = self.emit_expr(obj);
                // .len on arrays and strings
                if field == "len" {
                    match &obj_ty {
                        Type::Array(_) => {
                            let r = self.fresh_reg();
                            self.emit(&format!("  {} = call i64 @ionic_array_len(ptr {})", r, obj_val));
                            return (r, Type::Int64);
                        }
                        Type::Str => {
                            let r = self.fresh_reg();
                            self.emit(&format!("  {} = call i64 @strlen(ptr {})", r, obj_val));
                            return (r, Type::Int64);
                        }
                        _ => {}
                    }
                }
                if let Type::Struct(sname) = &obj_ty {
                    if let Some(layout) = self.structs.get(sname).cloned() {
                        if let Some(idx) = Self::struct_field_offset(&layout, field) {
                            let field_ty = layout.fields[idx].1.clone();
                            let llty = Self::llvm_ty(&field_ty);
                            let gep = self.fresh_reg();
                            let reg = self.fresh_reg();
                            self.emit(&format!(
                                "  {} = getelementptr i64, ptr {}, i64 {}",
                                gep, obj_val, idx
                            ));
                            self.emit(&format!("  {} = load {}, ptr {}, align 8", reg, llty, gep));
                            return (reg, field_ty);
                        }
                    }
                }
                ("0".to_string(), Type::Unknown)
            }

            Expr::Index { obj, idx } => {
                let (obj_val, obj_ty) = self.emit_expr(obj);
                let (idx_val, _)      = self.emit_expr(idx);
                let elem_ty = match &obj_ty {
                    Type::Array(inner) => *inner.clone(),
                    Type::Str => Type::Int64,
                    _ => Type::Unknown,
                };
                let raw = self.fresh_reg();
                self.emit(&format!(
                    "  {} = call i64 @ionic_array_get(ptr {}, i64 {})",
                    raw, obj_val, idx_val
                ));
                let reg = self.i64_to_elem(&raw, &elem_ty);
                (reg, elem_ty)
            }

            Expr::ArrayLit(elems) => {
                let cap = elems.len().max(8) as i64;
                let ptr = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @ionic_array_new(i64 8, i64 {})", ptr, cap));
                let mut elem_ty = Type::Unknown;
                for elem in elems {
                    let (v, ty) = self.emit_expr(elem);
                    if matches!(elem_ty, Type::Unknown) { elem_ty = ty.clone(); }
                    let i64_v = self.ptr_to_i64(&v, &ty);
                    self.emit(&format!("  call void @ionic_array_push(ptr {}, i64 {})", ptr, i64_v));
                }
                (ptr, Type::Array(Box::new(elem_ty)))
            }

            Expr::ToGpu(inner) => {
                let (reg, _) = self.emit_expr(inner);
                self.emit("  ; .toGpu() — Phase 1 stub");
                (reg, Type::Tensor(Box::new(HwTarget::Gpu)))
            }

            Expr::ToCpu(inner) => {
                let (reg, _) = self.emit_expr(inner);
                self.emit("  ; .toCpu() — Phase 1 stub");
                (reg, Type::Tensor(Box::new(HwTarget::Cpu)))
            }
        }
    }

    fn emit_call(&mut self, callee: &str, args: &[(String, Type)]) -> (String, Type) {
        match callee {
            "print" | "println" => {
                for (val, ty) in args {
                    self.emit_print_val(val, ty);
                }
                ("0".to_string(), Type::Void)
            }
            "exit" => {
                let (v, _) = &args[0];
                let t = self.fresh_reg();
                self.emit(&format!("  {} = trunc i64 {} to i32", t, v));
                self.emit(&format!("  call void @exit(i32 {})", t));
                ("0".to_string(), Type::Void)
            }
            "sqrt" => {
                let (v, _) = &args[0];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call double @sqrt(double {})", r, v));
                (r, Type::Float64)
            }
            "abs" => {
                let (v, _) = &args[0];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call double @fabs(double {})", r, v));
                (r, Type::Float64)
            }
            "int64_to_float64" => {
                let (v, _) = &args[0];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = sitofp i64 {} to double", r, v));
                (r, Type::Float64)
            }
            "float64_to_int64" => {
                let (v, _) = &args[0];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = fptosi double {} to i64", r, v));
                (r, Type::Int64)
            }
            "len" => {
                let (v, ty) = &args[0];
                let r = self.fresh_reg();
                match ty {
                    Type::Array(_) => self.emit(&format!("  {} = call i64 @ionic_array_len(ptr {})", r, v)),
                    Type::Str      => self.emit(&format!("  {} = call i64 @strlen(ptr {})", r, v)),
                    _ => self.emit(&format!("  {} = add i64 0, 0 ; len unsupported type", r)),
                }
                (r, Type::Int64)
            }
            "push" => {
                let (arr, _) = args[0].clone();
                let (val, vt) = args[1].clone();
                let i64_val = self.ptr_to_i64(&val, &vt);
                self.emit(&format!("  call void @ionic_array_push(ptr {}, i64 {})", arr, i64_val));
                ("0".to_string(), Type::Void)
            }
            "str_len" => {
                let (v, _) = &args[0];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call i64 @strlen(ptr {})", r, v));
                (r, Type::Int64)
            }
            "str_concat" => {
                let (a, _) = &args[0];
                let (b, _) = &args[1];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @ionic_str_concat(ptr {}, ptr {})", r, a, b));
                (r, Type::Str)
            }
            "str_eq" => {
                let (a, _) = &args[0];
                let (b, _) = &args[1];
                let cmp = self.fresh_reg();
                let r   = self.fresh_reg();
                self.emit(&format!("  {} = call i32 @strcmp(ptr {}, ptr {})", cmp, a, b));
                self.emit(&format!("  {} = icmp eq i32 {}, 0", r, cmp));
                let ext = self.fresh_reg();
                self.emit(&format!("  {} = zext i1 {} to i64", ext, r));
                (ext, Type::Int64)
            }
            "str_index" => {
                let (s, _) = &args[0];
                let (i, _) = &args[1];
                let r = self.fresh_reg();
                let gep = self.fresh_reg();
                self.emit(&format!("  {} = getelementptr i8, ptr {}, i64 {}", gep, s, i));
                let byte = self.fresh_reg();
                self.emit(&format!("  {} = load i8, ptr {}, align 1", byte, gep));
                self.emit(&format!("  {} = sext i8 {} to i64", r, byte));
                (r, Type::Int64)
            }
            "int64_to_str" => {
                let (v, _) = &args[0];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @ionic_int64_to_str(i64 {})", r, v));
                (r, Type::Str)
            }
            "float64_to_str" => {
                let (v, _) = &args[0];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @ionic_float64_to_str(double {})", r, v));
                (r, Type::Str)
            }
            "char_to_str" => {
                let (v, _) = &args[0];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @ionic_char_to_str(i64 {})", r, v));
                (r, Type::Str)
            }
            "get_arg" => {
                let (idx, _) = &args[0];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @ionic_get_arg(i64 {})", r, idx));
                (r, Type::Str)
            }
            "cpu_core_count" => {
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call i64 @ionic_cpu_core_count()", r));
                (r, Type::Int64)
            }
            "file_exists" => {
                let (path, _) = &args[0];
                let r32 = self.fresh_reg();
                self.emit(&format!("  {} = call i32 @access(ptr {}, i32 0)", r32, path));
                let cmp = self.fresh_reg();
                self.emit(&format!("  {} = icmp eq i32 {}, 0", cmp, r32));
                let r = self.fresh_reg();
                self.emit(&format!("  {} = zext i1 {} to i64", r, cmp));
                (r, Type::Int64)
            }
            "load_model" => {
                let (v, _) = &args[0];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @ionic_load_model(ptr {})", r, v));
                (r, Type::Model(Box::new(HwTarget::Cpu)))
            }
            "model_free" => {
                let (v, _) = &args[0];
                self.emit(&format!("  call void @ionic_model_free(ptr {})", v));
                ("0".to_string(), Type::Void)
            }
            "piper_forward" => {
                let (mdl, _) = &args[0];
                let (ph, _)  = &args[1];
                let (ns, _)  = &args[2];
                let (ls, _)  = &args[3];
                let (nw, _)  = &args[4];
                let r = self.fresh_reg();
                self.emit(&format!(
                    "  {} = call ptr @ionic_piper_forward(ptr {}, ptr {}, double {}, double {}, double {})",
                    r, mdl, ph, ns, ls, nw));
                (r, Type::Array(Box::new(Type::Float64)))
            }
            "write_wav" => {
                let (path, _) = &args[0];
                let (arr, _)  = &args[1];
                let (ns, _)   = &args[2];
                let (sr, _)   = &args[3];
                self.emit(&format!(
                    "  call void @ionic_write_wav(ptr {}, ptr {}, i64 {}, i64 {})",
                    path, arr, ns, sr));
                ("0".to_string(), Type::Void)
            }
            "gguf_generate" => {
                let (mdl, _)  = &args[0];
                let (prompt, _) = &args[1];
                let (max_tok, _) = &args[2];
                let r = self.fresh_reg();
                self.emit(&format!(
                    "  {} = call ptr @ionic_gguf_generate(ptr {}, ptr {}, i64 {})",
                    r, mdl, prompt, max_tok));
                (r, Type::Str)
            }
            "gguf_set_temp" => {
                let (mdl, _)  = &args[0];
                let (temp, _) = &args[1];
                self.emit(&format!("  call void @ionic_gguf_set_temp(ptr {}, double {})", mdl, temp));
                ("0".to_string(), Type::Void)
            }
            "gguf_set_top_p" => {
                let (mdl, _)   = &args[0];
                let (top_p, _) = &args[1];
                self.emit(&format!("  call void @ionic_gguf_set_top_p(ptr {}, double {})", mdl, top_p));
                ("0".to_string(), Type::Void)
            }

            // ── MLX tensor backend ────────────────────────────────────────────
            // Every mlx_* entry point takes only i64/ptr and returns i64/ptr,
            // so a single data-driven arm covers the whole surface. The return
            // type comes from the signature table (src/semantic.rs) rather than
            // a second hard-coded list, so adding a builtin there is enough to
            // make it callable here too.
            //
            // float64 crosses this boundary as an IEEE-754 *bit pattern* in an
            // i64 register, matching the native ARM64 codegen and the C
            // runtime's i642d()/d2i64() helpers. Emitting `double` instead
            // would put the value in a D register while the callee reads X —
            // the call returns garbage rather than failing, so the bitcast is
            // mandatory in both directions.
            name if name.starts_with("mlx_") => {
                // Signatures live in one place: semantic::mlx_builtins(). Both
                // this arm and the checker read that table, so a builtin cannot
                // be typed in one phase and unknown in the other.
                let ret_ty = self.builtins.get(name)
                    .map(|(_, r)| r.clone())
                    .unwrap_or(Type::Int64);
                let ret_is_void = ret_ty == Type::Void;
                let callee = format!("ionic_{}", name);

                let mut args_ir: Vec<String> = Vec::new();
                for (v, t) in args {
                    if *t == Type::Float64 {
                        // double -> i64 bit pattern
                        let bits = self.fresh_reg();
                        self.emit(&format!("  {} = bitcast double {} to i64", bits, v));
                        args_ir.push(format!("i64 {}", bits));
                    } else {
                        let llty = match t {
                            Type::Str | Type::Array(_) | Type::Tensor(_) | Type::Model(_) => "ptr",
                            _ => "i64",
                        };
                        args_ir.push(format!("{} {}", llty, v));
                    }
                }

                if ret_is_void {
                    self.emit(&format!("  call void @{}({})", callee, args_ir.join(", ")));
                    ("0".to_string(), Type::Void)
                } else if ret_ty == Type::Float64 {
                    // i64 bit pattern -> double, so the rest of codegen sees a
                    // normal float value.
                    let raw = self.fresh_reg();
                    self.emit(&format!("  {} = call i64 @{}({})", raw, callee, args_ir.join(", ")));
                    let r = self.fresh_reg();
                    self.emit(&format!("  {} = bitcast i64 {} to double", r, raw));
                    (r, ret_ty)
                } else {
                    let llret = Self::llvm_ty(&ret_ty).to_string();
                    let r = self.fresh_reg();
                    self.emit(&format!("  {} = call {} @{}({})", r, llret, callee, args_ir.join(", ")));
                    (r, ret_ty)
                }
            }
            "fgets_stdin" => {
                let (max_bytes, _) = &args[0];
                let buf = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @malloc(i64 {})", buf, max_bytes));
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @ionic_fgets_stdin(ptr {}, i32 {})", r, buf, max_bytes));
                (r, Type::Str)
            }
            "file_write_binary" => {
                let (arr, _) = &args[0];
                let (path, _) = &args[1];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call i64 @ionic_file_write_binary(ptr {}, ptr {})", r, arr, path));
                (r, Type::Int64)
            }
            "arr_reset" => {
                let (arr, _) = &args[0];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call i64 @ionic_arr_reset(ptr {})", r, arr));
                (r, Type::Int64)
            }
            "file_read" => {
                let (v, _) = &args[0];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @ionic_file_read(ptr {})", r, v));
                (r, Type::Str)
            }
            "file_write" => {
                let (path, _) = &args[0];
                let (content, _) = &args[1];
                let fp = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @fopen(ptr {}, ptr @mode_w)", fp, path));
                self.emit(&format!("  call i32 @fputs(ptr {}, ptr {})", content, fp));
                self.emit(&format!("  call i32 @fclose(ptr {})", fp));
                ("1".to_string(), Type::Int64)
            }
            "file_open_read" => {
                let (path, _) = &args[0];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @fopen(ptr {}, ptr @mode_r)", r, path));
                // Store ptr as i64 via ptrtoint
                let ri = self.fresh_reg();
                self.emit(&format!("  {} = ptrtoint ptr {} to i64", ri, r));
                (ri, Type::Int64)
            }
            "file_open_write" => {
                let (path, _) = &args[0];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @fopen(ptr {}, ptr @mode_w)", r, path));
                let ri = self.fresh_reg();
                self.emit(&format!("  {} = ptrtoint ptr {} to i64", ri, r));
                (ri, Type::Int64)
            }
            "file_close" => {
                let (fd, _) = &args[0];
                let fp = self.fresh_reg();
                self.emit(&format!("  {} = inttoptr i64 {} to ptr", fp, fd));
                self.emit(&format!("  call i32 @fclose(ptr {})", fp));
                ("0".to_string(), Type::Void)
            }
            "file_read_line" => {
                let (fd, _) = &args[0];
                let fp = self.fresh_reg();
                self.emit(&format!("  {} = inttoptr i64 {} to ptr", fp, fd));
                let buf = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @malloc(i64 4096)", buf));
                let ret = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @fgets(ptr {}, i32 4096, ptr {})", ret, buf, fp));
                (buf, Type::Str)
            }
            "file_write_line" => {
                let (fd, _)   = &args[0];
                let (line, _) = &args[1];
                let fp = self.fresh_reg();
                self.emit(&format!("  {} = inttoptr i64 {} to ptr", fp, fd));
                self.emit(&format!("  call i32 @fputs(ptr {}, ptr {})", line, fp));
                ("0".to_string(), Type::Void)
            }
            "file_eof" => {
                let (fd, _) = &args[0];
                let fp = self.fresh_reg();
                self.emit(&format!("  {} = inttoptr i64 {} to ptr", fp, fd));
                let r32 = self.fresh_reg();
                self.emit(&format!("  {} = call i32 @feof(ptr {})", r32, fp));
                let r = self.fresh_reg();
                self.emit(&format!("  {} = sext i32 {} to i64", r, r32));
                (r, Type::Int64)
            }
            "system" => {
                let (cmd, _) = &args[0];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call i64 @ionic_system(ptr {})", r, cmd));
                (r, Type::Int64)
            }
            // String / system builtins whose Ionic name maps to an `ionic_*`
            // runtime symbol. Needed to compile the self-hosted sources, which
            // use str_slice, getenv and target_is_linux internally.
            "str_slice" => {
                let (s, _) = &args[0];
                let (a, _) = &args[1];
                let (b, _) = &args[2];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @ionic_str_slice(ptr {}, i64 {}, i64 {})", r, s, a, b));
                (r, Type::Str)
            }
            "str_replace" => {
                let (s, _)     = &args[0];
                let (from, _)  = &args[1];
                let (to, _)    = &args[2];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @ionic_str_replace(ptr {}, ptr {}, ptr {})", r, s, from, to));
                (r, Type::Str)
            }
            "str_contains" => {
                let (s, _) = &args[0];
                let (sub, _) = &args[1];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call i64 @ionic_str_contains(ptr {}, ptr {})", r, s, sub));
                (r, Type::Int64)
            }
            "str_starts_with" => {
                let (s, _) = &args[0];
                let (p, _) = &args[1];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call i64 @ionic_str_starts_with(ptr {}, ptr {})", r, s, p));
                (r, Type::Int64)
            }
            "str_ends_with" => {
                let (s, _) = &args[0];
                let (p, _) = &args[1];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call i64 @ionic_str_ends_with(ptr {}, ptr {})", r, s, p));
                (r, Type::Int64)
            }
            "getenv" => {
                let (name, _) = &args[0];
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @ionic_getenv(ptr {})", r, name));
                (r, Type::Str)
            }
            "read_line" => {
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call ptr @ionic_read_line()", r));
                (r, Type::Str)
            }
            "str_to_int64" => {
                // Only rewrite to the runtime symbol when the program does not
                // define its own `str_to_int64` (the compiler sources do).
                if self.fns.contains_key("str_to_int64") {
                    let ret_ty = self.fns.get("str_to_int64").map(|(_, r)| r.clone()).unwrap();
                    let llret = Self::llvm_ty(&ret_ty).to_string();
                    let args_ir: Vec<String> = args.iter()
                        .map(|(v, t)| format!("{} {}", Self::llvm_ty(t), v)).collect();
                    let r = self.fresh_reg();
                    self.emit(&format!("  {} = call {} @str_to_int64({})", r, llret, args_ir.join(", ")));
                    (r, ret_ty)
                } else {
                    let (s, _) = &args[0];
                    let r = self.fresh_reg();
                    self.emit(&format!("  {} = call i64 @ionic_str_to_int64(ptr {})", r, s));
                    (r, Type::Int64)
                }
            }
            "target_is_linux" => {
                let r = self.fresh_reg();
                self.emit(&format!("  {} = call i64 @ionic_target_is_linux()", r));
                (r, Type::Int64)
            }
            other => {
                // User-defined or extern C function.
                // Unknown functions default to returning i64 so they can be used as arguments.
                let ret_ty = self.fns.get(other).map(|(_, r)| r.clone()).unwrap_or(Type::Int64);
                let llret  = Self::llvm_ty(&ret_ty).to_string();
                let args_ir: Vec<String> = args.iter().map(|(v, t)| {
                    format!("{} {}", Self::llvm_ty(t), v)
                }).collect();
                if ret_ty == Type::Void {
                    self.emit(&format!("  call void @{}({})", other, args_ir.join(", ")));
                    ("0".to_string(), Type::Void)
                } else {
                    let r = self.fresh_reg();
                    self.emit(&format!("  {} = call {} @{}({})", r, llret, other, args_ir.join(", ")));
                    (r, ret_ty)
                }
            }
        }
    }

    fn emit_print_val(&mut self, val: &str, ty: &Type) {
        match ty {
            Type::Int64 => {
                self.emit(&format!("  call i32 (ptr, ...) @printf(ptr @fmt_int, i64 {})", val));
            }
            Type::Float64 => {
                self.emit(&format!("  call i32 (ptr, ...) @printf(ptr @fmt_float, double {})", val));
            }
            Type::Str => {
                self.emit(&format!("  call i32 (ptr, ...) @printf(ptr @fmt_str, ptr {})", val));
            }
            _ => {
                // Treat as i64 fallback
                self.emit(&format!("  call i32 (ptr, ...) @printf(ptr @fmt_int, i64 {})", val));
            }
        }
    }
}

fn raw_byte_count(escaped: &str) -> usize {
    let mut count = 0;
    let bytes = escaped.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 2 < bytes.len() {
            count += 1;
            i += 3;
        } else {
            count += 1;
            i += 1;
        }
    }
    count
}
