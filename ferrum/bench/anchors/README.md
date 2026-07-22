# CCRL anchor pool

This directory defines the reproducible external-engine pool used by Ferrum's
anchored gauntlet. `bin/` is gitignored: built engine and Ordo binaries never
belong in git. This README, rather than checked-in binaries, is the source of
truth for rebuilding the pool.

From this directory, create the gitignored binary output directory before
building any anchors:

```bash
mkdir -p bin
```

The ratings below are the published single-CPU values from the
[CCRL 40/4 (Blitz) all-engines list](https://www.computerchess.org.uk/ccrl/404/rating_list_all.html),
captured on 2026-07-16 from the list whose last games were added on 2026-07-11.
They are the fixed anchor values intended for the version-specific Ordo setup
in Task 17. The installed pool spans 2296 to 3423 CCRL.

| Gauntlet name | Upstream version | Pinned commit | CCRL Blitz rating |
| --- | --- | --- | ---: |
| `stash-v17` | Stash 17.0 | `f85863d68b2f20812f8e1b330cd2613212a1b12a` | 2296 |
| `stash-v21` | Stash 21.0 | `6dc8c9cdefaa43e3d82a7c59cd2a5f93bcbd9d5a` | 2713 |
| `stash-v37` | Stash 37.0 | `077a93828fee354853ffac322f178d1a4f1670ff` | 3423 |
| `weiss-2.0` | Weiss 2.0 | `0db49c7270f5c6286a983390a5ea2780e1e0c9f4` | 3320 |

The CCRL list also has multi-CPU Weiss entries; 3320 is deliberately the
single-CPU entry because the gauntlet launches each anchor with its default one
thread.

## Build Stash anchors

Repository: <https://github.com/mhouppin/stash-bot>

Each Stash revision builds natively on Apple Silicon with the upstream
Makefile. `ARCH=generic` explicitly selects the non-x86 path. Use a clean clone
per revision so object files cannot cross version boundaries:

```bash
git clone https://github.com/mhouppin/stash-bot.git stash-v17-src
git -C stash-v17-src checkout --detach f85863d68b2f20812f8e1b330cd2613212a1b12a
make -C stash-v17-src/src ARCH=generic
cp stash-v17-src/src/stash-bot bin/stash-v17

git clone https://github.com/mhouppin/stash-bot.git stash-v21-src
git -C stash-v21-src checkout --detach 6dc8c9cdefaa43e3d82a7c59cd2a5f93bcbd9d5a
make -C stash-v21-src/src ARCH=generic
cp stash-v21-src/src/stash-bot bin/stash-v21

git clone https://github.com/mhouppin/stash-bot.git stash-v37-src
git -C stash-v37-src checkout --detach 077a93828fee354853ffac322f178d1a4f1670ff
make -C stash-v37-src/src ARCH=generic
cp stash-v37-src/src/stash bin/stash-v37
```

The commits correspond to the upstream `v17.0`, `v21.0`, and `v37.0` tags.

## Build Weiss 2.0

Repository: <https://github.com/TerjeKir/weiss>

Apple Clang diagnoses old-style C prototypes more strictly than the compiler
used for the release. The upstream Makefile promotes those diagnostics with
`-Werror`; overriding `WARN` keeps the warnings visible without making them
fatal. No source patch is required.

```bash
git clone https://github.com/TerjeKir/weiss.git weiss-2.0-src
git -C weiss-2.0-src checkout --detach 0db49c7270f5c6286a983390a5ea2780e1e0c9f4
make -C weiss-2.0-src/src basic WARN='-Wall -Wextra -Wshadow'
cp weiss-2.0-src/src/weiss bin/weiss-2.0
```

The commit corresponds to the upstream `v2.0` tag.

## Build Ordo

Repository: <https://github.com/michiguel/Ordo>

Ordo is pinned to `v1.2.6` at
`17eec774f2e4b9fdd2b1b38739f55ea221fb851a`. macOS does not provide
`pthread_spinlock_t`; the project's documented `NSPINLOCKS` switch maps those
locks to pthread mutexes.

```bash
git clone https://github.com/michiguel/Ordo.git ordo-src
git -C ordo-src checkout --detach 17eec774f2e4b9fdd2b1b38739f55ea221fb851a
make -C ordo-src clean
make -C ordo-src CFLAGS='-DNDEBUG -DMY_SEMAPHORES -DNSPINLOCKS -flto -I myopt -I sysport'
cp ordo-src/ordo bin/ordo
```

After building, mark every file in `bin/` executable and verify each engine
answers `uci` with `uciok` and `isready` with `readyok`. Verify the rating tool
with `bin/ordo -h`.

Task 3 intentionally stops at a rough pool estimate. `anchors.txt` and the
exact Ordo `--multi-anchors` invocation are fixed and version-checked in Task
17, when the M1 exit rating is measured.
