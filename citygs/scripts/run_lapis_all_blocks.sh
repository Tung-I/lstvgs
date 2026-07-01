#!/bin/bash
# Resumable, MULTI-SESSION-PARALLEL driver for the remaining 8 Lapis-CityGS blocks (block 4 done).
# Safe to run simultaneously from several GPU sessions/nodes — they self-distribute with no overlap:
#   * Cross-node coordination: a block is claimed by atomically `mkdir`-ing
#     results/rubble_lapis_block<NNN>/.train_claim on the shared /work fs. Whoever wins trains it;
#     others skip. The claim dir's `owner` file records host+pid+time (so you can see who has what).
#   * One-training-per-node: before claiming, wait until this node has no train_lapisgs.py running,
#     so a single GPU never runs two blocks at once.
#   * Skip-aware: a block is "done" iff layer_03_full.ply AND eval/metrics_fixed_factor4.json exist.
# Loops until ALL blocks are done, waiting out blocks in progress on other nodes. Just launch the
# SAME command in each session. Stale claim (node died mid-block)? remove the block's .train_claim.
# Usage: bash run_lapis_all_blocks.sh [STEPS_PER_LAYER]   (default 30000)
set -o pipefail
STEPS="${1:-30000}"
WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
LSTVGS="$WORK_DIR/lstvgs"
RESULTS="$LSTVGS/results"
LOG="$RESULTS/rubble_lapis_allblocks.log"
HOST="$(hostname)"
ORDER=(8 0 2 6 1 5 7 3)   # smallest->largest cam count; block 4 (747) already done

is_complete() { [ -f "$1/layers/layer_03_full.ply" ] && [ -f "$1/eval/metrics_fixed_factor4.json" ]; }

echo "=== driver START host=$HOST pid=$$ $(date); steps=$STEPS order=${ORDER[*]} ===" | tee -a "$LOG"
while true; do
    for B in "${ORDER[@]}"; do
        BID="$(printf '%03d' "$B")"
        RDIR="$RESULTS/rubble_lapis_block$BID"
        CLAIM="$RDIR/.train_claim"
        is_complete "$RDIR" && continue
        # one training per node: wait for THIS node's GPU to be free
        while pgrep -f 'train_lapisgs.py' >/dev/null 2>&1; do sleep 60; done
        is_complete "$RDIR" && continue   # re-check: another node may have finished it while we waited
        mkdir -p "$RDIR"
        if mkdir "$CLAIM" 2>/dev/null; then
            echo "$HOST $$ $(date)" > "$CLAIM/owner"
            echo "--- [$HOST] CLAIM+TRAIN block $B $(date) ---" | tee -a "$LOG"
            bash "$LSTVGS/citygs/scripts/run_lapis_block.sh" "$B" "$STEPS"; rc=$?
            rm -rf "$CLAIM"
            echo "--- [$HOST] block $B EXIT=$rc $(date) ---" | tee -a "$LOG"
        else
            echo "--- [$HOST] block $B claimed by $(cat "$CLAIM/owner" 2>/dev/null) — skip ---" | tee -a "$LOG"
        fi
    done
    alldone=1
    for B in "${ORDER[@]}"; do is_complete "$RESULTS/rubble_lapis_block$(printf '%03d' "$B")" || alldone=0; done
    [ "$alldone" -eq 1 ] && { echo "=== [$HOST] ALL blocks complete $(date) ===" | tee -a "$LOG"; break; }
    echo "--- [$HOST] pass complete; waiting on in-progress blocks elsewhere $(date) ---" | tee -a "$LOG"
    sleep 120
done
DONE=$(ls "$RESULTS"/rubble_lapis_block*/eval/metrics_fixed_factor4.json 2>/dev/null | wc -l)
echo "=== driver END host=$HOST $(date); blocks complete = $DONE/9 (incl block 4) ===" | tee -a "$LOG"
