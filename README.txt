### FANTASY PREMIER LEAGUE POINTS PREDICTIONS ###
This project is a work in progress!

The aim is to create a neural network model that predicts the total points a player will earn in the next FPL season. The data storage and model development will use Google Cloud platform services.

There are three components to this project:
- Retrieving historical season stats for each currently-active player through the FPL API.
- Developing a neural network model (keras) to predict the points earnt in the next season using the previous season's data, and training it on the Vertex AI platform.
- A linear optimisation solution to choose the highest-scoring team within the parameters of FPL (budget, team composition etc.).

Current progress:
- API requests complete. 
- API jsons have been stored as BigQuery objects and processed using SQL.
- A DNN has been built for the "midfielder" role, and it has been trained on the Google Cloud AI platform. This initial model has not been tuned at all yet, we will do that after the deployment pipeline is fully ready.
- The test model has been successfully deployed using the Google Cloud AI platform.

Next steps:
- Test accessing the model via the REST API.
- Tune the midfielder model.
- Repeat for other player roles (GK, DEF, FWD)
