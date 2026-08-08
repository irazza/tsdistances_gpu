# tsdistances_gpu — moved

**This repository is archived. Development continues in
[irazza/tsdistances](https://github.com/irazza/tsdistances), under
`crates/tsdistances_gpu/`.**

The crate was merged into the main `tsdistances` repository as a Cargo workspace
member on 2026-08-08 ([PR #4](https://github.com/irazza/tsdistances/pull/4)).
Its full history came along, so `git blame` and `git bisect` work there exactly
as they did here — the commit hashes are unchanged, because the merge grafted
this history in rather than rewriting it.

The last commit made here is `8479c12`.

## Why the move

`tsdistances` depended on this repository as a **git dependency**, and that
boundary had begun to cost more than it bought:

- **The nightly pin was coupled but unenforced.** Both crates pin the same Rust
  nightly, but a dependency's `rust-toolchain.toml` is ignored — only the root
  one applies. rust-gpu's `rustc_codegen_spirv` is a rustc backend built against
  an exact nightly and refuses to load against any other, so the two files had to
  agree, with nothing checking that they did.
- **This crate's build profile was silently dead.** Cargo ignores `[profile.*]`
  in non-root manifests, so the `[profile.release.build-override] opt-level = 3`
  here never took effect and `rustc_codegen_spirv` was compiled unoptimized on
  every cold build.
- **Every change was a three-step dance:** push here → `cargo update -p
  tsdistances_gpu` there → commit the lockfile. Cross-cutting changes could not
  be made or reviewed atomically.
- **This repository had no CI.** Nothing built or tested it on push; breakage
  surfaced downstream, later.

The two crates remain **separate crates** in the workspace, and always must:
this one compiles *itself* to SPIR-V (`build.rs` runs `SpirvBuilder::new(".")`,
`lib.rs` is `no_std` under `target_arch = "spirv"`), so it can never share a
compilation unit with PyO3, rayon or rustfft. Only the repository boundary was
removed.

## If you have an old checkout

Nothing breaks. This repository stays public and cloneable, so any
`Cargo.lock` pinning `git+https://github.com/irazza/tsdistances_gpu.git`
continues to resolve. It is simply frozen — new work goes to
[irazza/tsdistances](https://github.com/irazza/tsdistances).
