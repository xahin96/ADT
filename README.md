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
>
> python manage.py createsuperuser

# Go to 0.0.0.0:8000/datamining
# Go to 0.0.0.0:8000/datamining\load
**The landing page will show up**