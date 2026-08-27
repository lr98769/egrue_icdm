# for VARIABLE in 2025 2026 2027 2028
# do
#     echo "seed=$VARIABLE" > cur_seed.py
#     ipython --TerminalIPythonApp.file_to_run='2a_resnet18.ipynb'
# done

# for VARIABLE in 2025 2026 2027 2028
# do
#     echo "seed=$VARIABLE" > cur_seed.py
#     ipython --TerminalIPythonApp.file_to_run='2b_rue.ipynb'
# done

# for VARIABLE in 2025 2026 2027 2028
# do
#     echo "seed=$VARIABLE" > cur_seed.py
#     ipython --TerminalIPythonApp.file_to_run='2c_egrue.ipynb'
# done

# for VARIABLE in 2025 2026 2027 2028
# do
#     echo "seed=$VARIABLE" > cur_seed.py
#     ipython --TerminalIPythonApp.file_to_run='2d_postnet.ipynb'
# done

# for VARIABLE in 2025 2026 2027 2028
# do
#     echo "seed=$VARIABLE" > cur_seed.py
#     ipython --TerminalIPythonApp.file_to_run='2e_bnn.ipynb'
# done

# for VARIABLE in 2025 2026 2027 2028
# do
#     echo "seed=$VARIABLE" > cur_seed.py
#     ipython --TerminalIPythonApp.file_to_run='2f_del.ipynb'
# done

for VARIABLE in 2024 2025 2026 2027 2028
do
    echo "seed=$VARIABLE" > cur_seed.py
    ipython --TerminalIPythonApp.file_to_run='3_evaluate.ipynb'
done

