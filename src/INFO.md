> Код в процессе доработки

    cd src

Default (small, n_iterations_graph=2)

    python run_experiment.py

Choose experiment size

    python run_experiment.py experiment=medium
    python run_experiment.py experiment=large
    python run_experiment.py experiment=xlarge

Override any value on the fly

    python run_experiment.py experiment=large experiment.n_iterations_global=3
    python run_experiment.py device=cpu
    python run_experiment.py seed=42 experiment=small

Run all four sizes

    for exp in small medium large xlarge; do
        python run_experiment.py experiment=$exp
    done