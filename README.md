# ROBUSTNESS AND INTERPRETABILITY IN CONVOLUTIONAL NEURAL NETWORKS #

### Description ###
Codebase for the paper "ON ROBUSTNESS AND INTERPRETABILITY IN CONVOLUTIONAL NEURAL NETWORKS", a final year project done at NTU.

Robustness is assessed with adversarial accuracy using AutoAttack, while interpretability is evaluated based on the explanation quality of saliency maps - quantified using the ROAD AUC, Sparseness, and Max-Sensitivity metrics.

![cifar_qualitative drawio](https://github.com/user-attachments/assets/cc1b0abd-bc54-4d99-8096-edb99a211aa2)


### Running experiments ###
1. Edit `config.ini` to point to the directory used to store the CIFAR-10 and ImageNet datasets. For ImageNet, the `ILSVRC2012_devkit_t12.tar.gz`, `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` files should be downloaded from https://image-net.org/challenges/LSVRC/2012/2012-downloads.php and placed into this directory. 
2. Execute `pip install -r requirements.txt` in the root of this repository
3. Choose a configuration file from the `config` directory
3. Run `python3 main.py --config config/cifar10/resnet18/pgd.yml --test-standard --test-robust --test-interpretable`, passing in your chosen file as an arugment, to train and evaluate a model
4. The trained model, as well as as a log file showing the results, are saved to the `logs` directory. To load a previously trained model for evaluation, the path to the `.pt` file may be passed as a `--state` argument, based on the following example: 
`python3 main.py --config config/cifar10/resnet18/pgd.yml --state logs/cifar10/resnet18/pgd.yml/80a5c0a1_best_model_params.pt --test-standard --test-robust --test-interpretable`

