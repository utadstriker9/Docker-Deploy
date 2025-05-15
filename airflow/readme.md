Overview
========

Welcome to Astronomer! This project was generated after you ran 'astro dev init' using the Astronomer CLI. This readme describes the contents of the project, as well as how to run Apache Airflow on your local machine.
Project Contents
================

Your Astro project contains the following files and folders:

- dags: This folder contains the Python files for your Airflow DAGs. By default, this directory includes one ex
- `example_astronauts`: This DAG shows a simple ETL pipeline example that queries the list of astronauts cu
- Dockerfile: This file contains a versioned Astro Runtime Docker image that provides a differentiated Airflow 
- include: This folder contains any additional files that you want to include as part of your project. It is em
- packages.txt: Install OS-level packages needed for your project by adding them to this file. It is empty by d
- requirements.txt: Install Python packages needed for your project by adding them to this file. It is empty by
- plugins: Add custom or community plugins for your project to this file. It is empty by default.
- airflow_settings.yaml: Use this local-only file to specify Airflow Connections, Variables, and Pools instead
Deploy Your Project Locally
===========================

1. Start Airflow on your local machine by running 'astro dev start'.

This command will spin up 4 Docker containers on your machine, each for a different Airflow component:

- Postgres: Airflow's Metadata Database
- Webserver: The Airflow component responsible for rendering the Airflow UI
- Scheduler: The Airflow component responsible for monitoring and triggering tasks
- Triggerer: The Airflow component responsible for triggering deferred tasks

2. Verify that all 4 Docker containers were created by running 'docker ps'.

Note: Running 'astro dev start' will start your project with the Airflow Webserver exposed at port 8080 and Pos>
3. Access the Airflow UI for your local Airflow project. To do so, go to http://localhost:8080/ and log in with>
You should also be able to access your Postgres Database at 'localhost:5432/postgres'.

Deploy Your Project to Astronomer
=================================

If you have an Astronomer account, pushing code to a Deployment on Astronomer is simple. For deploying instructions, refer to Astronomer documentation: https://docs.astronomer.io/cloud/deploy-code/. 

This command will spin up 4 Docker containers on your machine, each for a different Airflow component:

- Postgres: Airflow's Metadata Database
- Webserver: The Airflow component responsible for rendering the Airflow UI
- Scheduler: The Airflow component responsible for monitoring and triggering tasks
- Triggerer: The Airflow component responsible for triggering deferred tasks

2. Verify that all 4 Docker containers were created by running 'docker ps'.                                     
Note: Running 'astro dev start' will start your project with the Airflow Webserver exposed at port 8080 and Pos

3. Access the Airflow UI for your local Airflow project. To do so, go to http://localhost:8080/ and log in with>You should also be able to access your Postgres Database at 'localhost:5432/postgres'.
                                                                                                                Deploy Your Project to Astronomer

Contact
=======

The Astronomer CLI is maintained with love by the Astronomer team. To report a bug or suggest a change, reach out to our support.