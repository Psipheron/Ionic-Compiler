;; Tree-sitter highlight queries for Ionic.
;;
;; Node names here come from grammar.js, which is generated from the language
;; the compiler itself accepts (src/lexer, src/parser). Validate with:
;;   cd tree-sitter-ionic && tree-sitter query queries/highlights.scm <file>
;;
;; Ordering note: when several patterns match the same node, the LAST matching
;; pattern wins, so the generic `(identifier) @variable` fallback is placed
;; early and the more specific identifier captures follow it.

;; ============================================================================
;; Keywords & Control Flow
;; ============================================================================
[
  "fn"
  "let"
  "mut"
  "return"
  "import"
  "struct"
] @keyword

[
  "if"
  "else"
  "while"
  "for"
  "in"
  "break"
  "continue"
] @keyword.control

;; @cpu / @gpu placement annotations (@gpu(0.5) included)
(placement_tag) @keyword.modifier

;; ============================================================================
;; Literals
;; ============================================================================
(int_literal) @number
(float_literal) @number
(bool_literal) @constant.builtin

(string_literal) @string
(comment) @comment @spell

;; ============================================================================
;; Operators & Punctuation
;; ============================================================================
[
  "=" "+=" "-=" "*=" "/="
  "+" "-" "*" "/" "%"
  "==" "!=" "<" ">" "<=" ">="
  "&&" "||" "&" "|" "^" "<<" ">>" "!" "~"
] @operator

[
  "->"
  ".."
] @operator

[
  "(" ")"
  "{" "}"
  "[" "]"
] @punctuation.bracket

[
  ";"
  ","
  "."
  ":"
] @punctuation.delimiter

;; ============================================================================
;; Generic identifier fallback (must precede the specific captures below)
;; ============================================================================
(identifier) @variable

;; ============================================================================
;; Types
;; ============================================================================
(primitive_type) @type.builtin

(tensor_type
  "tensor" @type.builtin)

(tensor_type
  ["@cpu" "@gpu"] @keyword.modifier)

;; `model@cpu` / `model@gpu` are placement types too; without these the tag on a
;; model type highlighted as nothing.
(model_type
  "model" @type.builtin)

(model_type
  ["@cpu" "@gpu"] @keyword.modifier)

(named_type
  (identifier) @type)

(struct_declaration
  name: (identifier) @type)

;; ============================================================================
;; Functions & Calls
;; ============================================================================
(function_declaration
  name: (identifier) @function)

(call_expression
  function: (identifier) @function.call)

(method_call
  method: (identifier) @function.method)

(conversion_expression
  method: ["toGpu" "toCpu"] @function.method)

;; Standard-library / runtime builtins (kept in sync with
;; src/semantic/checker.ionic register_builtins)
((identifier) @function.builtin
  (#any-of? @function.builtin
    "print" "println" "format" "exit" "system"
    "int64_to_str" "float64_to_str" "char_to_str" "str_to_int64"
    "str_concat" "str_len" "str_eq" "str_index" "str_slice" "str_replace"
    "str_contains" "str_starts_with" "str_ends_with" "str_hash"
    "int64_to_float64" "float64_to_int64"
    "sqrt" "fabs" "pow" "floor" "ceil" "min" "max" "abs"
    "len" "push" "arr_reset"
    "get_arg" "getenv" "read_line"
    "file_read" "file_write" "file_write_binary"
    "cpu_core_count" "target_is_linux"
    "http_get" "http_post" "http_status" "http_body" "http_urlencode"
    "load_model" "model_free" "model_forward" "gguf_generate"
    "gguf_set_temp" "gguf_set_top_p" "piper_forward" "write_wav"))

;; ============================================================================
;; Fields & properties
;; ============================================================================
(field_declaration
  name: (identifier) @property)

(field_initializer
  name: (identifier) @property)

(field_access
  field: (identifier) @property)

;; ============================================================================
;; Imports
;; ============================================================================
(import_declaration
  (string_literal) @string.special.path)

(import_declaration
  (dotted_path) @module)

(import_list
  (identifier) @module)
