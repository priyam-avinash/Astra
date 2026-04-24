#!/bin/bash
# ASTRA Weekly Model Retraining — local cron
# Schedule: every Sunday at 2am IST (30 20 * * 6 UTC)
# Crontab entry: 30 20 * * 6 /Users/avinashpriyam/Desktop/trading-app/react-algo-trading-app/backend/retrain_weekly.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/Users/avinashpriyam/.gemini/antigravity/scratch/algo-trading-app/backend/.venv/bin/python"
LOG_FILE="$SCRIPT_DIR/logs/retrain_$(date +%Y%m%d).log"
SUMMARY_FILE="$SCRIPT_DIR/logs/last_retrain_summary.txt"

mkdir -p "$SCRIPT_DIR/logs"

echo "========================================" | tee -a "$LOG_FILE"
echo "ASTRA Retrain: $(date '+%Y-%m-%d %H:%M:%S IST')" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Run training, capturing full output
cd "$SCRIPT_DIR" && $PYTHON train_models.py 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# ── Extract key metrics from log ──────────────────────────────────────
{
echo ""
echo "========================================"
echo "ASTRA Training Summary"
echo "Run: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# RF OOB score (logged by sklearn as oob_score_)
RF_OOB=$(grep -oP "OOB[^:]*:\s*\K[0-9]+\.[0-9]+" "$LOG_FILE" | tail -1)
[ -n "$RF_OOB" ] && echo "RF OOB R²          : $RF_OOB" \
                 || echo "RF OOB R²          : (not found in log)"

# LSTM val_loss and val_accuracy (Keras epoch logs)
LSTM_VAL_LOSS=$(grep -oP "val_loss: \K[0-9]+\.[0-9]+" "$LOG_FILE" | tail -1)
LSTM_VAL_ACC=$(grep -oP "val_accuracy: \K[0-9]+\.[0-9]+" "$LOG_FILE" | tail -1)
[ -n "$LSTM_VAL_LOSS" ] && echo "LSTM val_loss      : $LSTM_VAL_LOSS" \
                        || echo "LSTM val_loss      : (not found)"
[ -n "$LSTM_VAL_ACC"  ] && echo "LSTM val_accuracy  : $LSTM_VAL_ACC" \
                        || echo "LSTM val_accuracy  : (not found)"

# Count successes / failures from train_models.py summary block
SUCCESSES=$(grep -c "SUCCESS" "$LOG_FILE" 2>/dev/null || echo 0)
FAILURES=$(grep -c "FAILED\|❌" "$LOG_FILE" 2>/dev/null || echo 0)
echo "Models succeeded   : $SUCCESSES"
echo "Models failed      : $FAILURES"

echo ""
echo "-- Failures / Errors --"
ERRORS=$(grep -E "ERROR|FAILED|❌|Exception|Traceback" "$LOG_FILE" | head -20)
[ -n "$ERRORS" ] && echo "$ERRORS" || echo "(none)"

echo ""
echo "Full log: $LOG_FILE"
} | tee -a "$LOG_FILE" | tee "$SUMMARY_FILE"

# Re-read FAILURES for the notification (subshell above lost the var)
FAILURES=$(grep -c "FAILED\|❌" "$LOG_FILE" 2>/dev/null || echo 0)
RF_OOB=$(grep -oP "OOB[^:]*:\s*\K[0-9]+\.[0-9]+" "$LOG_FILE" | tail -1)
LSTM_VAL_LOSS=$(grep -oP "val_loss: \K[0-9]+\.[0-9]+" "$LOG_FILE" | tail -1)

if [ "$FAILURES" = "0" ] && [ $EXIT_CODE -eq 0 ]; then
    echo "" && echo "✅ All models retrained successfully. Restart backend to load new weights."
else
    echo "" && echo "⚠️  Training completed with issues — see $LOG_FILE"
fi

# ── macOS Notification Centre ─────────────────────────────────────────
if command -v osascript &>/dev/null; then
    if [ "$FAILURES" = "0" ] && [ $EXIT_CODE -eq 0 ]; then
        osascript -e "display notification \"All models retrained ✅  RF OOB: ${RF_OOB:-N/A}  LSTM val_loss: ${LSTM_VAL_LOSS:-N/A}\" with title \"ASTRA Weekly Retrain\" subtitle \"$(date '+%a %d %b %H:%M')\""
    else
        osascript -e "display notification \"${FAILURES} model(s) failed — check logs\" with title \"ASTRA Weekly Retrain ⚠️\" subtitle \"$(date '+%a %d %b %H:%M')\""
    fi
fi
