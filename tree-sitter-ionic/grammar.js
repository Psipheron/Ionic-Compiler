// Tree-sitter grammar for Ionic, derived from the real lexer/parser
// (src/lexer/tokens.ionic, src/lexer/lexer.ionic, src/parser/parser.ionic).
//
// It exists so editor queries (queries/highlights.scm) can be validated against
// an actual parser instead of guessed node names. Keep it in sync with the
// language as defined by the compiler itself.

const PREC = {
  OR: 1,
  AND: 2,
  BITOR: 3,
  BITXOR: 4,
  BITAND: 5,
  EQ: 6,
  CMP: 7,
  SHIFT: 8,
  ADD: 9,
  MUL: 10,
  UNARY: 11,
  POSTFIX: 12,
};

module.exports = grammar({
  name: 'ionic',

  extras: $ => [
    /[ \t\r\n]/,
    $.comment,
  ],

  word: $ => $.identifier,

  rules: {
    source_file: $ => repeat($._item),

    _item: $ => choice(
      $.import_declaration,
      $.struct_declaration,
      $.function_declaration,
      $.statement,
    ),

    // ── Comments ──────────────────────────────────────────────────────────
    comment: _ => token(choice(
      seq('//', /.*/),
      seq('/*', /[^*]*\*+([^/*][^*]*\*+)*/, '/'),
    )),

    // ── Imports ───────────────────────────────────────────────────────────
    import_declaration: $ => seq(
      'import',
      choice(
        $.string_literal,                       // import "path/file.ionic";
        seq($.dotted_path, optional(choice(
          seq('.', '*'),                        // import std.math.*;
          seq('.', $.import_list),              // import std.str.{contains, trim};
        ))),
      ),
      ';',
    ),
    dotted_path: $ => prec.right(1, seq($.identifier, repeat(seq('.', $.identifier)))),
    import_list: $ => seq('{', commaSep1($.identifier), '}'),

    // ── Placement tags ────────────────────────────────────────────────────
    placement_tag: $ => seq(
      choice('@gpu', '@cpu'),
      optional(seq('(', $.float_literal, ')')),
    ),

    // ── Structs ───────────────────────────────────────────────────────────
    struct_declaration: $ => seq(
      'struct',
      field('name', $.identifier),
      '{',
      commaSep($.field_declaration),
      '}',
    ),
    field_declaration: $ => seq(
      field('name', $.identifier),
      ':',
      field('type', $.type),
    ),

    // ── Functions ─────────────────────────────────────────────────────────
    function_declaration: $ => seq(
      optional($.placement_tag),
      'fn',
      field('name', $.identifier),
      field('parameters', $.parameter_list),
      optional(seq('->', field('return_type', $.type))),
      field('body', $.block),
    ),
    parameter_list: $ => seq('(', commaSep($.parameter), ')'),
    parameter: $ => seq(
      optional($.placement_tag),
      field('type', $.type),
      field('name', $.identifier),
    ),

    // ── Types ─────────────────────────────────────────────────────────────
    type: $ => choice(
      $.primitive_type,
      $.tensor_type,
      $.model_type,
      $.array_type,
      $.named_type,
    ),
    primitive_type: _ => choice('int64', 'float64', 'bool', 'string', 'void'),
    tensor_type: $ => seq('tensor', optional(choice('@cpu', '@gpu'))),
    // `model@cpu` / `model@gpu` are placement types, exactly like `tensor@cpu`
    // — `parse_type()` in src/parser/parser.ionic reads the tag the same way
    // for both keywords.
    //
    // `model` used to sit in primitive_type with no tag, so `model@cpu` parsed
    // as `model` followed by a stray `@cpu` and the file reported an ERROR.
    // That is why examples/mastery/04_placement_types.ionic never parsed.
    model_type: $ => seq('model', optional(choice('@cpu', '@gpu'))),
    array_type: $ => seq('[', $.type, ']'),
    named_type: $ => $.identifier,

    // ── Statements ────────────────────────────────────────────────────────
    statement: $ => choice(
      $.let_declaration,
      $.assignment_statement,
      $.expression_statement,
      $.return_statement,
      $.break_statement,
      $.continue_statement,
      $.if_statement,
      $.while_statement,
      $.for_statement,
      $.gpu_block,
      $.block,
    ),

    let_declaration: $ => seq(
      optional($.placement_tag),
      choice('let', 'mut'),
      field('name', $.identifier),
      optional(seq(':', field('type', $.type))),
      '=',
      field('value', $._expression),
      ';',
    ),

    assignment_statement: $ => seq(
      field('left', $._expression),
      field('operator', choice('=', '+=', '-=', '*=', '/=')),
      field('right', $._expression),
      ';',
    ),

    expression_statement: $ => seq($._expression, ';'),
    return_statement: $ => seq('return', optional($._expression), ';'),
    break_statement: _ => seq('break', ';'),
    continue_statement: _ => seq('continue', ';'),

    if_statement: $ => seq(
      'if', '(', field('condition', $._expression), ')',
      field('consequence', $.block),
      optional(seq('else', field('alternative', choice($.block, $.if_statement)))),
    ),
    while_statement: $ => seq(
      'while', '(', field('condition', $._expression), ')',
      field('body', $.block),
    ),
    for_statement: $ => seq(
      'for', '(',
      field('variable', $.identifier), 'in',
      field('start', $._expression), '..', field('end', $._expression),
      ')',
      field('body', $.block),
    ),
    gpu_block: $ => seq('gpu', $.block),
    block: $ => seq('{', repeat($.statement), '}'),

    // ── Expressions ───────────────────────────────────────────────────────
    _expression: $ => choice(
      $.identifier,
      $.int_literal,
      $.float_literal,
      $.string_literal,
      $.bool_literal,
      $.binary_expression,
      $.unary_expression,
      $.call_expression,
      $.method_call,
      $.field_access,
      $.index_expression,
      $.array_literal,
      $.struct_literal,
      $.parenthesized_expression,
      $.conversion_expression,
    ),

    parenthesized_expression: $ => seq('(', $._expression, ')'),

    binary_expression: $ => choice(
      ...binary($, '||', PREC.OR),
      ...binary($, '&&', PREC.AND),
      ...binary($, '|', PREC.BITOR),
      ...binary($, '^', PREC.BITXOR),
      ...binary($, '&', PREC.BITAND),
      ...binary($, '==', PREC.EQ),
      ...binary($, '!=', PREC.EQ),
      ...binary($, '<', PREC.CMP),
      ...binary($, '>', PREC.CMP),
      ...binary($, '<=', PREC.CMP),
      ...binary($, '>=', PREC.CMP),
      ...binary($, '<<', PREC.SHIFT),
      ...binary($, '>>', PREC.SHIFT),
      ...binary($, '+', PREC.ADD),
      ...binary($, '-', PREC.ADD),
      ...binary($, '*', PREC.MUL),
      ...binary($, '/', PREC.MUL),
      ...binary($, '%', PREC.MUL),
    ),

    unary_expression: $ => prec(PREC.UNARY, seq(
      field('operator', choice('-', '!', '~')),
      field('operand', $._expression),
    )),

    call_expression: $ => prec(PREC.POSTFIX, seq(
      field('function', $.identifier),
      field('arguments', $.argument_list),
    )),
    method_call: $ => prec(PREC.POSTFIX, seq(
      field('receiver', $._expression),
      '.',
      field('method', $.identifier),
      field('arguments', $.argument_list),
    )),
    field_access: $ => prec(PREC.POSTFIX, seq(
      field('receiver', $._expression),
      '.',
      field('field', $.identifier),
    )),
    index_expression: $ => prec(PREC.POSTFIX, seq(
      field('receiver', $._expression),
      '[', field('index', $._expression), ']',
    )),
    conversion_expression: $ => prec(PREC.POSTFIX, seq(
      field('receiver', $._expression),
      '.',
      field('method', choice('toGpu', 'toCpu')),
      '(', ')',
    )),

    argument_list: $ => seq('(', commaSep($._expression), ')'),
    array_literal: $ => seq('[', commaSep($._expression), ']'),
    struct_literal: $ => seq(
      field('type', $.identifier),
      '{', commaSep($.field_initializer), '}',
    ),
    field_initializer: $ => seq(
      field('name', $.identifier),
      ':',
      field('value', $._expression),
    ),

    // ── Literals & identifiers ────────────────────────────────────────────
    bool_literal: _ => choice('true', 'false'),
    int_literal: _ => token(choice(
      /[0-9]+/,
      /0[xX][0-9a-fA-F]+/,
    )),
    float_literal: _ => token(seq(/[0-9]+/, '.', /[0-9]+/)),
    string_literal: _ => token(seq('"', repeat(choice(/[^"\\]/, /\\./)), '"')),

    identifier: _ => /[a-zA-Z_][a-zA-Z0-9_]*/,
  },
});

// All binary operators are left-associative in the real parser (each level is
// a `while` loop that folds left), so the grammar mirrors that.
function binary($, operator, precedence) {
  return [prec.left(precedence, seq(
    field('left', $._expression),
    field('operator', operator),
    field('right', $._expression),
  ))];
}

function commaSep(rule) {
  return optional(commaSep1(rule));
}

function commaSep1(rule) {
  return seq(rule, repeat(seq(',', rule)));
}
