# HAC framework

This is the repository for the HAC framework for hierarchical publication classification. As a case study, this experiment uses publications from the Fiocruz at different CNPq levels.

## Project Structure

The project is organized as follows:

- **input/**: Contains the input data for the project.
- **models/**: Stores the trained models.
- **notebooks/**: Jupyter notebooks used for data analysis and model training.
- **results/**: Stores model performance results.
- **src/**: Contains the project's source code.
- **LICENSE**: GNU v3 license file.
- **README.md**: This file.

## How to Use

To use this project, follow these steps:

1. Clone this repository to your local environment.
2. Install the required dependencies listed in `requirements.txt`.
3. Use the notebooks in the `notebooks/` folder to explore the data and train models.
4. Store trained models in the `models/` folder.
5. Make predictions using the models and store the results in the `results/` folder.

## Environment Setup

To set up the development environment for this project, follow the instructions below:

1. **Install Miniconda**

   Before starting, ensure you have Miniconda installed. You can download and install Miniconda from the official website: [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Creating the Conda environment**

   After installing Miniconda, create a new Conda environment with Python 3.11 by running the following command in your terminal:

   ```bash
   conda create -n hac python=3.11 -y
   ```

3. **Activating the environment**

   Next, activate the newly created Conda environment by running:

   ```bash
   conda activate hac
   ```

4. **Installing packages**

   Now, install all the necessary dependencies for this project by running:

   ```bash
   pip install -r requirements.txt
   ```

5. **Running the Notebooks**

   You can start exploring the data and training models by running the notebooks in the notebooks/ folder. To launch Jupyter Notebook, simply type the following command in the terminal:

   ```bash
   jupyter notebook
   ```

## Contributions

Contributions are welcome! If you would like to contribute to this project, follow these steps:

1. Fork this repository.
2. Create a new branch for your feature (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add a new feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Merge Request on GitHub.

## License

This project is licensed under GNU v3. See the `LICENSE` file for more details.
