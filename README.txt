### FANTASY PREMIER LEAGUE POINTS PREDICTIONS ###
This project is a work in progress!

The aim is to create a neural network model that predicts the total points a player will earn in the next FPL season. Due to the amount of data that will be required, the data storage and model development will use Google Cloud platform services.

There are three components to this project:
- Retrieving historical season stats for each currently-active player through the FPL API.
- Developing a neural network model (keras) to predict the points earnt in the next season using the previous season's data, and training it on the Vertex AI platform.
- A linear optimisation solution to choose the highest-scoring team within the parameters of FPL (budget, team composition etc.).