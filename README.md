python3 -m http.server 3000

https://launch.smarthealthit.org

http://localhost:80/launch.html


docker image build -t flask_docker .
docker run -p 4000:4000 -p 3000:3000 -d flask_docker
