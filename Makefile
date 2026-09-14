.PHONY: all layer1 layer2 layer3 models figures clean

all: layer1 layer2 layer3 models figures

layer1:
	python -m layer1_signal_estimation.run

layer2:
	python -m layer2_anomaly_detection.run

layer3:
	python -m layer3_causal_analysis.run

models:
	python -m model_cross_validation.run_all

figures:
	python results/figures/scripts/fig01_channel_scope.py
	python results/figures/scripts/fig02_model_cross_validation.py
	python results/figures/scripts/fig03_quasi_experimental.py
	python results/figures/scripts/fig04_scm_and_heterogeneity.py
	python results/figures/scripts/fig05_dataset_and_precedence.py

clean:
	rm -rf results/figures/*.png
