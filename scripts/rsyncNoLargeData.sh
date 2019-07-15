echo MAGI: $HOST_MAGI_core_I port $HOST_MAGI_core_I_port
SRC="$HOST_MAGI_core_I:/home/beantowel/FDU/MIR/MirLabs/Lab_MelExt/data/"
DST="/home/beantowel/FDU/MIR/MirLabs/Lab_MelExt/data/"
rsync -auz -v -e "ssh -p $HOST_MAGI_core_I_port" --exclude='*.pkl' "$SRC" "$DST"
