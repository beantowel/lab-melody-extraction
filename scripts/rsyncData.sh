echo MAGI: $HOST_MAGI_core_I port $HOST_MAGI_core_I_port
rsync -auz -v -e "ssh -p $HOST_MAGI_core_I_port"  $HOST_MAGI_core_I:/home/beantowel/FDU/MIR/MirLabs/Lab_MelExt/data/ /home/beantowel/FDU/MIR/MirLabs/Lab_MelExt/data/
