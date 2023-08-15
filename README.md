# Descrizione pipeline di allenamento:

il file main.py (che potremmo rinominare train.py) crea una cartella, output_dir, per ogni esperimento, che contiene i checkpoint, i plot, i log di allenamento e i risultati dei test. Gli hyperparametri sono segnati nel file params.txt, per non doverli mettere tutti nei nomi dei file. Manca ancora l'implementazione del flag udr e l'allenamento a partire da un checkpoint.

argomenti da linea di comando:
output-dir, test-eps, lr, timesteps, use_udr, n-distr, source-env, target-env, checkpoint
aggiungibili:
batch_size policy:[MLP, CNN], overwrite (per i test)

Struttura della cartella:
output-dir/
    checkpoints/
        source_checkpoint.pt   #serve callback, per allenamenti lunghi, con due sottocartelle
        target_checkpoint.pt
    train_logs
        source_train_log.csv    #rinomina, per source e target
        target_train_log.csv
    plots/                    #distinguere source e plot anche qui
        source_train_plot.svg
        target_train_plot.svg
