# How to commit & push this repo (with the `gsplat` submodule)

This repo (`lstvgs`) contains a **git submodule** at `gsplat/` (see `.gitmodules`).
A submodule is a *separate* git repo nested inside this one. The parent repo does
**not** store the submodule's files — it only stores a pointer (one commit SHA)
to a specific commit in the submodule's own repo.

That has two consequences you must keep in mind:

1. Editing files **inside `gsplat/`** does nothing to the parent until you commit
   *inside* the submodule first, then commit the moved pointer in the parent.
2. `git status` in the parent shows the submodule as a single line
   (`modified: gsplat (new commits)`), not the individual file changes.

Below is the exact, repeatable flow. The "code/notes only" path is what you'll
use 95% of the time, because we normally don't touch `gsplat/`.

---

## 0. Where am I, and what changed?

```bash
cd /work/pi_rsitaram_umass_edu/tungi/lstvgs

git status                 # parent-repo changes (submodule shows as one line)
git submodule status       # SHA + name of each submodule; '+' prefix = pointer moved
```

- A leading `+` in `git submodule status` means the submodule is checked out at a
  *different* commit than what the parent records (pointer moved).
- A leading `-` means the submodule isn't checked out at all.
- No prefix = pointer matches; nothing to do for the submodule.

---

## 1. The common case: commit parent changes only (no `gsplat/` edits)

### 1a. Watch out for huge files BEFORE you stage

`results/` can hold gigabytes of PLYs, PNGs, and logs. `.gitignore` filters most
of it, but **always verify the staged size before committing** — GitHub rejects
any single file >100 MB and a bloated history is painful to undo.

```bash
# Stage in two steps so the big tree is filtered by .gitignore, not force-added:
git add .gitignore citygs/ notes/        # fast: the code/notes
git add results/                         # may take a minute; .gitignore filters it

# VERIFY what you're about to commit:
git diff --cached --name-only | wc -l                          # how many files
git diff --cached --name-only -z | xargs -0 du -bc | tail -1   # total bytes

# Flag anything > 1 MB (should normally be empty):
git diff --cached --name-only | while read f; do
  sz=$(stat -c%s "$f" 2>/dev/null)
  [ "${sz:-0}" -gt 1048576 ] && echo "$((sz/1024/1024))MB  $f"
done
```

If a big file slipped through (e.g. a loose `merged.ply`, raw `images_4_png/`, or
a multi-MB `*.log`), **don't commit** — add a rule to `.gitignore`, then restage:

```bash
# edit .gitignore to add the pattern, e.g.  results/**/*.ply
git reset                                # unstage everything
git add .gitignore citygs/ notes/ && git add results/   # restage cleanly
# ...re-run the verify checks above...
```

### 1b. Commit and push

```bash
git commit -m "Short summary line

- bullet of what changed
- bullet of what changed"

git push origin main
```

That's it for the everyday case.

---

## 2. The other case: you ALSO changed files inside `gsplat/`

You must commit **inside the submodule first**, then record the new pointer in the
parent. Order matters — if you push the parent before pushing the submodule,
collaborators get a pointer to a commit that doesn't exist on the remote.

```bash
# --- Step 1: commit & push INSIDE the submodule ---
cd /work/pi_rsitaram_umass_edu/tungi/lstvgs/gsplat
git status                       # these are the submodule's own changes
git add -A
git commit -m "gsplat: describe the change"
git push origin main             # push the submodule's repo FIRST

# --- Step 2: record the moved pointer in the PARENT ---
cd /work/pi_rsitaram_umass_edu/tungi/lstvgs
git status                       # now shows: modified: gsplat (new commits)
git add gsplat                   # stages the new submodule SHA pointer
git add .gitignore citygs/ notes/ results/   # any parent changes too
git commit -m "Bump gsplat + <your parent changes>"
git push origin main             # push the parent SECOND
```

Rule of thumb: **push the submodule before the parent.**

---

## 3. Fresh clone / new machine: pull the submodule too

A plain `git clone` leaves `gsplat/` empty. Either:

```bash
git clone --recurse-submodules https://github.com/Tung-I/lstvgs.git
```

or, if you already cloned:

```bash
git submodule update --init --recursive
```

To later fast-forward the submodule to whatever commit the parent points at:

```bash
git pull                                    # update parent (moves the pointer)
git submodule update --init --recursive     # check submodule out to that pointer
```

---

## Quick reference

| Goal | Commands |
|------|----------|
| See parent changes | `git status` |
| See submodule pointer state | `git submodule status` |
| Stage safely | `git add .gitignore citygs/ notes/ && git add results/` |
| Check staged size | `git diff --cached --name-only -z \| xargs -0 du -bc \| tail -1` |
| Commit + push parent | `git commit -m "..." && git push origin main` |
| Commit submodule | `cd gsplat && git add -A && git commit -m "..." && git push && cd ..` |
| Record moved pointer | `git add gsplat && git commit -m "Bump gsplat" && git push` |
| Clone with submodule | `git clone --recurse-submodules <url>` |
| Init submodule after clone | `git submodule update --init --recursive` |

**Golden rules**
1. Verify staged size before every commit — never let a PLY/PNG/big-log through.
2. Push the submodule before the parent.
3. `.gitignore` patterns must match the actual path: `results/**/ply/` does **not**
   catch a loose `results/foo/merged.ply` — use `results/**/*.ply`.
