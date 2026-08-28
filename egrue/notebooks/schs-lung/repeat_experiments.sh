# for VARIABLE in 2025 2026 2027 2028 2029 2030 2031 2032 2033
# do
#     echo "seed=$VARIABLE" > seed_file.py
#     ipython --TerminalIPythonApp.file_to_run='2g_de.ipynb'
# done


for VARIABLE in 2024 2025 2026 2027 2028 2029 2030 2031 2032 2033
do
    echo "seed=$VARIABLE" > seed_file.py
    ipython --TerminalIPythonApp.file_to_run='3_evaluate.ipynb'
done

