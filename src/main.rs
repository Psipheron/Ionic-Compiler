mod lexer;
mod ast;
mod parser;
mod imports;
mod resolve;
mod semantic;
mod codegen;

use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().collect();
    let (source_path, flags) = parse_args(&args);

    let mut source = fs::read_to_string(&source_path)
        .unwrap_or_else(|e| die(&format!("Cannot read '{}': {}", source_path, e)));

    // ── Relative-import splicing ──────────────────────────────────────────────
    // A source built from a tree of modules (`import "path.ionic";`) is
    // assembled depth-first into one buffer before lexing — the same way the
    // self-hosted compiler's `imp_expand` preprocesses its own sources. Plain
    // single-file programs are passed through untouched so their line numbers
    // stay intact for diagnostics.
    if resolve::has_relative_import(&source) {
        source = resolve::expand_entry(&source_path)
            .unwrap_or_else(|e| die(&format!("[Import Error] {}", e)));
    }

    // ── Lex ──────────────────────────────────────────────────────────────────
    let mut lexer = lexer::Lexer::new(&source);
    let tokens = lexer.tokenize().unwrap_or_else(|e| die(&format!("[Lex Error] {}", e)));

    if flags.dump_tokens {
        for t in &tokens {
            eprintln!("  [{:>3}:{:<3}] {:?}", t.line, t.col, t.node);
        }
        return;
    }

    // ── Parse ─────────────────────────────────────────────────────────────────
    let mut parser = parser::Parser::new(tokens);
    let mut program = parser.parse().unwrap_or_else(|e| die(&format!("[Parse Error] {}", e)));

    // ── Import resolution (selective compilation) ─────────────────────────────
    if !program.imports.is_empty() {
        let pool = imports::build_lib_pool(&program.imports)
            .unwrap_or_else(|e| die(&format!("[Import Error] {}", e)));
        let (lib_fns, lib_structs, lib_globals) = imports::bfs_reachable(&program, &pool);
        // Prepend library symbols so they appear before user code
        let mut merged_fns = lib_fns;
        merged_fns.extend(program.fns);
        program.fns = merged_fns;
        let mut merged_structs = lib_structs;
        merged_structs.extend(program.structs);
        program.structs = merged_structs;
        let mut merged_top = lib_globals;
        merged_top.extend(program.top_level);
        program.top_level = merged_top;
    }

    if flags.dump_ast {
        eprintln!("{:#?}", program);
        return;
    }

    // ── Semantic analysis ────────────────────────────────────────────────────
    let mut sem = semantic::SemanticAnalyzer::new();
    let errors = sem.analyze(&program);
    if !errors.is_empty() {
        for e in &errors {
            eprintln!("[Type Error] {}", e);
        }
        std::process::exit(1);
    }

    // ── Codegen ───────────────────────────────────────────────────────────────
    let mut cg = codegen::Codegen::new();
    let ir = cg.generate(&program);

    if flags.emit_ir {
        print!("{}", ir);
        return;
    }

    // ── Compile IR → native binary ────────────────────────────────────────────
    let stem = Path::new(&source_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("out");

    let out_bin = flags.output.clone().unwrap_or_else(|| stem.to_string());
    let ir_path = format!("/tmp/{}.ll", stem);

    fs::write(&ir_path, &ir)
        .unwrap_or_else(|e| die(&format!("Cannot write IR: {}", e)));

    let clang = find_clang();

    // Compile the model runtime C file if present
    let runtime_c   = runtime_c_path();
    let runtime_obj  = "/tmp/ionic_model_runtime.o";

    // Backend detection. Only link a backend the program actually references:
    // linking unconditionally would make every Ionic binary (including
    // hello-world) depend on mlx/llama/onnx and pay their init cost at startup.
    // Mirrors the same usage gate in src/main.ionic.
    let src = fs::read_to_string(&source_path).unwrap_or_default();
    let uses_mlx   = src.contains("mlx_");
    let uses_llama = src.contains("gguf_") || src.contains("load_model");
    let uses_ort   = src.contains("piper_forward")
                     || src.contains("model_forward")
                     || src.contains("load_model");

    let ort_inc  = if uses_ort { ort_include_path() } else { None };
    let ort_lib  = if uses_ort { ort_lib_path() } else { None };
    let has_ort  = ort_inc.is_some() && ort_lib.is_some();

    let llama_inc = if uses_llama { llama_include_path() } else { None };
    let llama_lib = if uses_llama { llama_lib_path() } else { None };
    let has_llama = llama_inc.is_some() && llama_lib.is_some();

    // MLX is Apple-silicon only. IONIC_NO_MLX=1 forces it off for a
    // half-installed mlx-c that would otherwise fail the link.
    let mlx_allowed = cfg!(target_os = "macos")
        && std::env::var("IONIC_NO_MLX").map(|v| v != "1").unwrap_or(true);
    let mlx_inc = if uses_mlx && mlx_allowed { mlx_include_path() } else { None };
    let mlx_lib = if uses_mlx && mlx_allowed { mlx_lib_path() } else { None };
    let has_mlx = mlx_inc.is_some() && mlx_lib.is_some();

    let has_runtime = if let Some(ref rc) = runtime_c {
        let mut compile_args: Vec<String> = vec![
            "-c".to_string(), rc.clone(), "-o".to_string(), runtime_obj.to_string(),
        ];
        if has_ort {
            compile_args.push(format!("-DIONIC_HAVE_ORT"));
            if let Some(ref inc) = ort_inc {
                compile_args.push(format!("-I{}", inc));
            }
        }
        if has_llama {
            compile_args.push(format!("-DIONIC_HAVE_LLAMA"));
            if let Some(ref inc) = llama_inc {
                compile_args.push(format!("-I{}", inc));
            }
        }
        if has_mlx {
            compile_args.push(format!("-DIONIC_HAVE_MLX"));
            if let Some(ref inc) = mlx_inc {
                compile_args.push(format!("-I{}", inc));
            }
        }
        let st = Command::new(&clang)
            .args(&compile_args)
            .status()
            .unwrap_or_else(|e| die(&format!("Cannot compile model runtime ({}): {}", rc, e)));
        st.success()
    } else {
        false
    };

    let mut clang_args: Vec<String> = vec![
        ir_path.clone(), "-o".to_string(), out_bin.clone(), "-lm".to_string(),
    ];
    if has_runtime {
        clang_args.push(runtime_obj.to_string());
    }
    if has_ort {
        if let Some(ref lib) = ort_lib {
            clang_args.push(format!("-L{}", lib));
            clang_args.push("-lonnxruntime".to_string());
            clang_args.push(format!("-Wl,-rpath,{}", lib));
        }
    }
    if has_llama {
        if let Some(ref lib) = llama_lib {
            clang_args.push(format!("-L{}", lib));
            clang_args.push("-lllama".to_string());
            clang_args.push(format!("-Wl,-rpath,{}", lib));
        }
    }
    if has_mlx {
        if let Some(ref lib) = mlx_lib {
            // libmlxc records its own dependencies by absolute path, so no
            // rpath entry is needed here.
            clang_args.push(format!("-L{}", lib));
            clang_args.push("-lmlxc".to_string());
        }
    }

    let status = Command::new(&clang)
        .args(&clang_args)
        .status()
        .unwrap_or_else(|e| die(&format!("Cannot run clang ({}): {}", clang, e)));

    if !status.success() {
        eprintln!("[Link Error] clang failed. IR saved at {}", ir_path);
        std::process::exit(1);
    }

    if flags.run {
        let status = Command::new(format!("./{}", out_bin))
            .status()
            .unwrap_or_else(|e| die(&format!("Cannot run binary: {}", e)));
        std::process::exit(status.code().unwrap_or(0));
    }

    eprintln!("Compiled: {} -> ./{}", source_path, out_bin);
}

struct Flags {
    dump_tokens: bool,
    dump_ast: bool,
    emit_ir: bool,
    run: bool,
    output: Option<String>,
}

fn parse_args(args: &[String]) -> (String, Flags) {
    if args.len() < 2 {
        eprintln!("ionic - Ionic compiler v0.1.0");
        eprintln!("");
        eprintln!("Usage: ionic <file.ionic> [options]");
        eprintln!("");
        eprintln!("Options:");
        eprintln!("  -o <name>      Output binary name");
        eprintln!("  --run          Compile and immediately execute");
        eprintln!("  --emit-ir      Print LLVM IR to stdout, do not compile");
        eprintln!("  --dump-ast     Print AST to stderr, do not compile");
        eprintln!("  --dump-tokens  Print token stream to stderr, do not compile");
        std::process::exit(0);
    }

    let source = args[1].clone();
    let mut flags = Flags {
        dump_tokens: false,
        dump_ast: false,
        emit_ir: false,
        run: false,
        output: None,
    };

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--dump-tokens" => flags.dump_tokens = true,
            "--dump-ast"    => flags.dump_ast = true,
            "--emit-ir"     => flags.emit_ir = true,
            "--run"         => flags.run = true,
            "-o" => {
                i += 1;
                flags.output = args.get(i).cloned();
            }
            unknown => {
                eprintln!("Unknown flag: {}", unknown);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    (source, flags)
}

/// Find ionic_model_runtime.c relative to the compiler binary or source tree.
fn runtime_c_path() -> Option<String> {
    // 1. Next to the compiler binary
    if let Ok(exe) = std::env::current_exe() {
        let candidate = exe.parent()?.join("ionic_model_runtime.c");
        if candidate.exists() {
            return Some(candidate.to_string_lossy().into_owned());
        }
    }
    // 2. In the source tree (dev mode)
    let candidate = Path::new("src/ionic_model_runtime.c");
    if candidate.exists() {
        return Some(candidate.to_string_lossy().into_owned());
    }
    None
}

fn find_clang() -> String {
    for candidate in &[
        "/opt/homebrew/opt/llvm/bin/clang",
        "/usr/local/opt/llvm/bin/clang",
        "clang",
    ] {
        if Command::new(candidate).arg("--version").output().is_ok() {
            return candidate.to_string();
        }
    }
    "clang".to_string()
}

fn die(msg: &str) -> ! {
    eprintln!("{}", msg);
    std::process::exit(1);
}

/// Return the llama.cpp include directory if found.
fn llama_include_path() -> Option<String> {
    let candidates = [
        "/opt/homebrew/include",
        "/opt/homebrew/Cellar/llama.cpp/9260/include",
        "/usr/local/include",
    ];
    for c in &candidates {
        if Path::new(c).join("llama.h").exists() {
            return Some(c.to_string());
        }
    }
    None
}

/// Return the llama.cpp library directory if found.
fn llama_lib_path() -> Option<String> {
    let candidates = [
        "/opt/homebrew/lib",
        "/opt/homebrew/Cellar/llama.cpp/9260/lib",
        "/usr/local/lib",
    ];
    for c in &candidates {
        if Path::new(c).join("libllama.dylib").exists()
            || Path::new(c).join("libllama.so").exists()
        {
            return Some(c.to_string());
        }
    }
    None
}

/// Return the MLX (C API) include directory if found. `#include "mlx/c/mlx.h"`
/// resolves from the prefix root, not from a `mlx/` subdirectory.
fn mlx_include_path() -> Option<String> {
    let candidates = [
        "/opt/homebrew/include",
        "/usr/local/include",
    ];
    for c in &candidates {
        if Path::new(c).join("mlx/c/mlx.h").exists() {
            return Some(c.to_string());
        }
    }
    None
}

/// Return the MLX (C API) library directory if found.
fn mlx_lib_path() -> Option<String> {
    let candidates = [
        "/opt/homebrew/lib",
        "/usr/local/lib",
    ];
    for c in &candidates {
        if Path::new(c).join("libmlxc.dylib").exists() {
            return Some(c.to_string());
        }
    }
    None
}

/// Return the ONNX Runtime include directory if found.
fn ort_include_path() -> Option<String> {
    let candidates = [
        "/opt/homebrew/include/onnxruntime",
        "/opt/homebrew/Cellar/onnxruntime/1.26.0/include/onnxruntime",
        "/usr/local/include/onnxruntime",
    ];
    for c in &candidates {
        if Path::new(c).exists() {
            return Some(c.to_string());
        }
    }
    None
}

/// Return the ONNX Runtime library directory if found.
fn ort_lib_path() -> Option<String> {
    let candidates = [
        "/opt/homebrew/lib",
        "/opt/homebrew/Cellar/onnxruntime/1.26.0/lib",
        "/usr/local/lib",
    ];
    for c in &candidates {
        let lib = Path::new(c).join("libonnxruntime.dylib");
        if lib.exists() {
            return Some(c.to_string());
        }
    }
    None
}
