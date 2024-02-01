https://launch.smarthealthit.org

http://localhost:3000/launch.html


docker image build -t flask_docker .
docker run -p 4000:4000 -p 3000:3000 -d flask_docker
