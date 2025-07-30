SHELL=/bin/bash
SERVICE=bwd_tek
VERSION=1.0.0
MYSQL_ROOT_PASSWORD=s3cr3t


build:
	docker build -f Dockerfile -t ai-proficient-mysql:${SERVICE}-${VERSION} .

run:
	docker run --rm --name aip-mysql-${SERVICE} -v $(CURDIR):/data \
		-p 3306:3306 \
		-e MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD} \
		ai-proficient-mysql:${SERVICE}-${VERSION}

exec:
	docker exec -it \
		-e MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD} \
		aip-mysql-${SERVICE} bash

insert:
	docker exec \
		-e MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD} \
		aip-mysql-${SERVICE} bash -c \
	                   "mysql -u root --password=${MYSQL_ROOT_PASSWORD} -D results_db < /data/insert.sql "

test:
	docker exec \
		-e MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD} \
		aip-mysql-${SERVICE} bash -c \
	                   "mysql -u root --password=${MYSQL_ROOT_PASSWORD} -D results_db < /data/test.sql "


stop:
	docker stop aip-mysql-${SERVICE}


clean:
	docker rmi ai-proficient-mysql:${SERVICE}-${VERSION}

