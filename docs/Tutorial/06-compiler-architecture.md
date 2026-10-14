# 06 — Compiler Architecture

**Goal:** read the compiler as an engineer. Know each stage, its IR, its source
file, and where to intervene to change behaviour. You will learn the two
structure-of-arrays representations (tokens and AST) and the exact place the
string-typing information is lost.

---

## 1. The pipeline

`src/main.ionic` is the driver:

```
read argv ─► imp_expand (splice imports) ─► lex_global
   ─► parse_program           (src/parser/parser.ionic)
   ─► sc_program              (src/semantic/checker.ionic)
   ─► opt_program             (src/opt/opt.ionic)
   ─► codegen_program         (src/codegen/elf.ionic → native.ionic)
   ─► system("clang <out>.o src/ionic_model_runtime.c -o <out>")
```

Read `src/main.ionic` fully — it is short and it *is* the architecture.

---

## 2. Token representation (structure of arrays)

The lexer pushes into five parallel global arrays rather than a list of structs:

- `g_tok_kinds[]` — token kind (`TOK_*`)
- `g_tok_ivals[]` — integer payload
- `g_tok_svals[]` — string payload
- `g_tok_lines[]`, `g_tok_cols[]` — source position

Token kinds are plain ints in `src/lexer/tokens.ionic` (`TOK_LET = 1`, …). The
lexer is a single-pass, one-char-lookahead scanner (`lx_peek2`) in
`src/lexer/lexer.ionic`; `keyword_kind(string)` maps identifiers to keyword
kinds; `@` has its own branch that maps `gpu`/`cpu` to `TOK_AT_GPU`/`TOK_AT_CPU`.

---

## 3. AST representation (structure of arrays)

Same idea, for nodes. Every node is an **integer index** into global arrays
(`src/parser/ast.ionic`):

- `nd_kind[i]` — `ND_*` node kind
- `nd_sv[i]` — string payload (identifier, field name, callee, …)
- `nd_iv[i]` — integer payload (literal value, op kind, mutability, …)
- `nd_i2[i]`, `nd_i3[i]`, `nd_i4[i]` — up to three child indices
- `nd_line[i]`, `nd_col[i]` — position, stamped from `g_cur_line`/`g_cur_col`
- `ND_CONS` — a cons-cell list (`i2`=head, `i3`=tail) for arg/element lists

Node kinds: `ND_INT`, `ND_FLOAT`, `ND_STR`, `ND_BOOL`, `ND_IDENT`, `ND_BINOP`,
`ND_UNOP`, `ND_CALL`, `ND_METHOD_CALL`, `ND_FIELD_ACCESS`, `ND_INDEX`,
`ND_ARRAY_LIT`, `ND_STRUCT_LIT`, `ND_FIELD_INIT`, `ND_TO_GPU`, `ND_TO_CPU`,
statements `ND_LET`…`ND_BLOCK`, definitions `ND_FN_DEF`/`ND_PARAM`/`ND_STRUCT_DEF`,
type nodes `ND_TY_*`, operator constants `OP_*` (including the float variants
`OP_FADD`…). Read `src/parser/ast.ionic` completely.

---

## 4. Parser: recursive descent, left-recursion removed

`src/parser/parser.ionic` implements a PRECEDENCE CLIMB as one function per level:
`parse_or → parse_and → parse_bitor → parse_bitxor → parse_bitand → parse_eq →
parse_cmp → parse_shift → parse_add → parse_mul → parse_unary → parse_postfix →
parse_primary`. Each `{ op operand }` repetition is an iteration, preserving
left-associativity. See `docs/homework_solutions.md` HW 2.1.

Compound assignment is **desugared here**: `x += e` becomes `ND_ASSIGN(x,
ND_BINOP(+, x, e))`. See `is_compound_assign` / `compound_assign_op`.

Error recovery: `p_sync_stmt` (statement level) and `p_sync_top` (definition
level) advance to the next reliable boundary; `g_parse_error` gates `p_expect` so
multiple diagnostics are produced in one pass.

---

## 5. Semantic checker: inference + placement

`src/semantic/checker.ionic` (~1300 lines). The core is `sc_expr(node)` setting
the globals `sc_ty`, `sc_sv`, `sc_elem_ty`. It:

- infers types (unit 02 §2) and **rewrites `OP_*` to `OP_F*`** on mixed
  int/float arithmetic;
- models the environment as a flat stack with frame markers
  (`sv_declare`/`sv_lookup_ty`/`sv_push_frame`/`sv_pop_frame`); lookup scans
  backwards so innermost scope wins;
- enforces placement via `sc_in_gpu_block` and `sc_in_conversion` (unit 04):
  - `sc_in_gpu_block == 1 && sc_ty == TY_TENSOR_CPU` → error;
  - `sc_in_gpu_block == 0 && sc_ty == TY_TENSOR_GPU` → error;
  - `sc_in_gpu_block == 1 && is_cpu_only_fn(callee)` → error;
  - `sc_in_gpu_block == 1 && hw_ann_is_cpu(fn_hws[fi])` → error.
- `register_builtins()` (line ~638) is the canonical list of every builtin and
  its arity/return type — commit it to memory for extending the language.

---

## 6. Optimizer: one pass, always semantics-preserving

`src/opt/opt.ionic`, entry `opt_program(fns, top)`. Passes (see `README.md`
"Optimizations" and the `opt_*` functions):

- **Constant folding** — int ops incl. comparisons + bitwise
  (`opt_fold_int`); float ops and single-arg float builtins computed as IEEE bit
  patterns (`opt_fold_float`, `opt_float_unop`).
- **Algebraic simplification + DCE** — `x+0`, `x*1`, `x/1`, dead `if` branches,
  unreachable-after-return (`opt_alg`).
- **Loop unrolling** — `for` with constant bounds and trip ≤ 8, no
  break/continue; body cloned and induction var substituted
  (`opt_try_unroll`-style, `opt_subst_var`).
- **Leaf inlining** — functions whose body is exactly `return <expr>` with no
  calls (`opt_collect_inlineables`, `opt_try_inline`, `opt_subst_general`).
- **Scoped const propagation** — `let x = <literal>` never reassigned is replaced
  by the literal (`opt_fn` prescan).

The optimizer must preserve the self-hosting fixed point, so it is deliberately
conservative.

---

## 7. Codegen: direct ARM64, spills everything

`src/codegen/native.ionic` does a **one-pass AST→ARM64** translation. It does
**not** build SSA (unit 09 explains the SSA design exercise). Strategy:

- params → callee-saved `X19..`;
- locals → spill slots at `[X29 + offset*8]`;
- scratch temporaries → `X9..` (`scratch_push`/`scratch_pop`), with explicit
  spill/restore around calls.

Read: `src/codegen/native.ionic` for `ncg_expr`, `local_find`, `local_get_str`,
`local_is_str`, the print dispatch (§ below), and `ND_STRUCT_LIT`.

`src/codegen/elf.ionic` selects native vs ELF output (`write_elf`).

---

## 8. THE central codegen wrinkle: `ncg_last_ty`

Printing is overloaded by argument type at codegen. `ncg_last_ty` (0=int, 1=str,
2=float, `native.ionic:31`) records the type of the most recently evaluated
expression so `print`/`println` can pick `ionic_print*` vs `ionic_println*` vs
the `_float64` variant.

**The problem:** `ncg_last_ty` is not set on every expression path.

- `ND_INDEX` (array element load, `native.ionic:1159`) computes the element and
  returns, **without setting `ncg_last_ty`**. So `println(arr[i])` on a
  `[string]` array falls through to the int64 path and prints the pointer.
- `ND_LET` with a non-literal string initializer doesn't always record string.
- Shadowing by name table (`local_find`/`local_get_str` history) caused further
  misclassification.

This one missing piece of information is simultaneously:

1. the cause of the string-array print bug (unit 02 §3.2), and
2. the **blocker** for the RC memory-model slice (unit 07 §5).

It is the single highest-leverage fix in the compiler: a uniform static type
oracle carried into codegen.

---

## 9. Diagnostics

`src/diagnostics.ionic` renders multi-error, careted, colored reports with
`path:line:col` jump links, enclosing-function context, and Levenshtein
"did you mean" hints. It honors `NO_COLOR`/`TERM=dumb`.

---

## Exercises

**06.1 — Trace the driver.** Annotate `src/main.ionic` line by line with which
stage each block invokes. Reference the exact function names.

**06.2 — Follow one token.** For a snippet containing `+=` and `tensor@gpu`,
list the token kinds the lexer emits (consult `tokens.ionic`) and the AST kinds
the parser produces (consult `ast.ionic`).

**06.3 — Find the desugaring.** Quote the lines in `src/parser/parser.ionic` that
turn `x += e` into an assignment.

**06.4 — Optimizer proof.** Take a `for (i in 0..6)` loop with a constant body
and show, via `--llvm` on the legacy path or by reasoning from `opt.ionic`, that
it is unrolled. Then add a `break` and show it is not unrolled.

**06.5 — The missing oracle.** Quote the `ND_INDEX` arm of `ncg_expr`
(`native.ionic:~1159`) and point to the exact line where `ncg_last_ty` *should*
be set for a string element but isn't. This is your capstone target.

**06.6 — Builtin registry.** Add a trivial no-op idea for a new builtin on paper:
which two files (`checker.register_builtins`, `native.ionic` call lowering, plus
the C runtime) must change? Don't implement yet — that's the capstone.

---

## Checkpoint 06

- [ ] You can draw the pipeline from argv to linked binary.
- [ ] You can explain both SoA representations and why `ND_CONS` exists.
- [ ] You can name all five optimizer passes and their entry functions.
- [ ] You can state, precisely, what `ncg_last_ty` is and where it is wrong.
- [ ] You can point to the exact file:line for placement enforcement.

## [Next](07-memory-model-and-runtime.md)
