docker-compose logs api > api.txt 2>&1
docker-compose logs worker > worker.txt 2>&1
docker-compose logs rabbitmq > rabbitmq.txt 2>&1
docker-compose logs redis > redis.txt 2>&1
docker-compose logs flower > flower.txt 2>&1
docker-compose logs db > db.txt 2>&1
docker-compose logs prometheus > prometheus.txt 2>&1
docker-compose logs grafana > grafana.txt 2>&1
docker-compose exec db psql -U postgres -d multiagent

rm -f api.txt worker.txt rabbitmq.txt redis.txt flower.txt db.txt prometheus.txt grafana.txt

hrough Docker. Here are the commands to help you:

To list all tables in the database:

bashCopydocker-compose exec db psql -U postgres -d multiagent -c "\dt"

To see the structure of the results table:

bashCopydocker-compose exec db psql -U postgres -d multiagent -c "\d results"

To view the contents of the results table:

docker-compose exec db psql -U postgres -d multiagent -c "SELECT * FROM results;"

If you want to limit the results to just a few rows:

bashCopydocker-compose exec db psql -U postgres -d multiagent -c "SELECT * FROM results LIMIT 5;"

If you want to see the most recent entries:

bashCopydocker-compose exec db psql -U postgres -d multiagent -c "SELECT * FROM results ORDER BY created_at DESC LIMIT 5;"

To check just specific columns:

bashCopydocker-compose exec db psql -U postgres -d multiagent -c "SEL


# First copy the migration script to the container
docker cp mn.py multiagent-api:/app/

# Then execute it
docker-compose exec api python /app/mn.py

docker-compose logs worker > work.txt
