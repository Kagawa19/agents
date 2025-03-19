docker-compose logs api > api.txt 2>&1
docker-compose logs worker > worker.txt 2>&1
docker-compose logs rabbitmq > rabbitmq.txt 2>&1
docker-compose logs redis > redis.txt 2>&1
docker-compose logs flower > flower.txt 2>&1
docker-compose logs db > db.txt 2>&1
docker-compose logs prometheus > prometheus.txt 2>&1
docker-compose logs grafana > grafana.txt 2>&1


rm -f api.txt worker.txt rabbitmq.txt redis.txt flower.txt db.txt prometheus.txt grafana.txt
