# Startup Commands:
>docker compose build

>docker compose up

# Migrate DB
**use this command to find the container id of web::**

>docker ps

**copy the container id**

**run this to log in**

>docker exec -it <container_id> /bin/bash

**run this to perform migrations**

> python manage.py makemigrations

> python manage.py migrate

# Go to 0.0.0.0:8000/datamining
**The landing page will show up**

---
### run on local machine

0. start the docker containers from `lcoal` folder by using `docker-compose up`
> if you have postgres installed locally on your machine, then just change the values in `adt/.env`

1. install dependencies `pip install -r requirements.txt`
2. Run these commands
   1. `python manage.py makemigrations`
   2. `python manage.py migrate`
3. Run the application using `python manage.py runserver 0.0.0.0:8000`
