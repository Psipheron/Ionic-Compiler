class Ionic < Formula
  desc "Statically-typed compiled language for data science and AI workloads"
  homepage "https://github.com/Psipheron/Ionic-Compiler"
  # No leading `v` here: Homebrew's audit forbids it (rubocops/version.rb) and
  # Version.new("v0.3.1") != Version.new("0.3.1"), so the keg would be named
  # `v0.3.1`.
  #
  # The tag *does* have the `v`, so the URL below cannot interpolate `version`
  # directly or it would ask for .../download/0.3.1/ionic-0.3.1-... and 404.
  # The prefix is re-added explicitly.
  version "0.3.1"
  license "MIT"

  # One architecture. The release workflow builds exactly
  # `ionic-<version>-aarch64-apple-darwin.tar.gz` and nothing else, so an
  # `on_intel` branch here could only ever point at a URL that is never
  # produced — with a literal `PLACEHOLDER_X86_SHA256` for a checksum. That made
  # `brew install` on an Intel Mac a 404 instead of an honest "not supported",
  # and the arm64 placeholder meant the arm64 branch could not install either.
  #
  # This is the reference copy; the live formula is in the tap repository and is
  # rewritten there on every stable release. Keep the two shapes identical or
  # the tap's rewrite step will not apply.
  depends_on arch: :arm64
  depends_on macos: :sonoma

  url "https://github.com/Psipheron/Ionic-Compiler/releases/download/v#{version}/ionic-v#{version}-aarch64-apple-darwin.tar.gz"
  sha256 "cdf3e95783a669e1da4970d6dcb106e0d9204a7b61ae4c583729a743b24a785d"

  def install
    # The self-hosted `ionic` locates its bundled standard library and C
    # runtime relative to its own executable directory ($IONIC_ROOT, set by
    # the C runtime from realpath(_NSGetExecutablePath)). Those assets must
    # therefore sit *next to* the real binary — not under `share/`.
    #
    # We install everything into `libexec/` and put a symlink on the PATH.
    # `realpath` resolves the symlink to libexec/ionic, so $IONIC_ROOT becomes
    # <prefix>/libexec and `<root>/lib/std/*.ionic` and
    # `<root>/src/ionic_model_runtime.c` resolve correctly from any CWD.
    libexec.install "ionic"
    libexec.install "lib"
    libexec.install "src"          # the tarball's src/ holds only the C runtime
    bin.install_symlink libexec/"ionic"

    doc.install "README.md"
    # The formula declares `license "MIT"`; Homebrew expects the notice to be
    # installed alongside the software it covers.
    doc.install "LICENSE"
  end

  def caveats
    <<~EOS
      The Ionic standard library and native runtime ship next to the compiler:

        #{opt_libexec}/lib
        #{opt_libexec}/src/ionic_model_runtime.c

      The `ionic` command is a symlink in #{HOMEBREW_PREFIX}/bin that points
      into libexec/. It resolves its own assets regardless of the current
      working directory, so you can compile from anywhere.

      ML backends are optional and are linked into the programs you compile,
      not into `ionic` itself. Install whichever you need:

        brew install mlx-c        # Apple-silicon GPU tensor runtime (mlx_* builtins)
        brew install onnxruntime  # ONNX inference + Piper TTS (piper_forward)

      Backends are linked only when a program actually references them, so a
      plain `ionic hello.ionic` stays dependency-free.

      MLX needs Apple silicon and Metal; it is unavailable on Intel Macs and on
      Linux. On those platforms `mlx_*` programs still compile and link, and
      report "MLX backend not compiled in" at runtime. ONNX Runtime is the only
      model-file backend on every target.

      Overrides, for a non-Homebrew prefix:
        IONIC_MLX_INCLUDE=/opt/mlx/include IONIC_MLX_LIB=/opt/mlx/lib \
            ionic model.ionic -o model

      To force a backend off (e.g. a half-installed mlx-c would fail to link):
        IONIC_NO_MLX=1 ionic model.ionic -o model
    EOS
  end

  test do
    (testpath/"hello.ionic").write <<~EOF
      fn main() -> int64 {
        print("hello from ionic");
        return 0;
      }
    EOF
    # The MLX surface must at least link on every platform: on Apple silicon it
    # runs, elsewhere it degrades to "not compiled in". Either way the compiled
    # program must build and exit 0.
    (testpath/"mlx_smoke.ionic").write <<~EOF
      fn main() -> int64 {
        if (mlx_available() == 0) { return 0; }
        let a = mlx_zeros([2, 2], 0);
        mlx_free(a);
        return 0;
      }
    EOF
    system bin/"ionic", "mlx_smoke.ionic", "-o", testpath/"mlx_smoke"
    system testpath/"mlx_smoke"

    # Run from the throwaway test dir (not the repo) so the repo-relative
    # fallback for lib/ and src/ionic_model_runtime.c cannot mask a broken
    # $IONIC_ROOT layout.
    testpath.cd do
      system bin/"ionic", "hello.ionic", "-o", testpath/"hello"
      assert_equal "hello from ionic", shell_output("#{testpath}/hello").strip

      # Guard the standalone layout: compiling an std.* program must resolve
      # both the bundled stdlib and the C runtime via $IONIC_ROOT.
      (testpath/"std_test.ionic").write <<~EOF
        import std.math;
        fn main() -> int64 {
          print(str_concat("sqrt2=", float64_to_str(sqrt(4.0))));
          return 0;
        }
      EOF
      system bin/"ionic", "std_test.ionic", "-o", testpath/"std_test"
      assert_equal "sqrt2=2", shell_output("#{testpath}/std_test").strip
    end
  end
end
