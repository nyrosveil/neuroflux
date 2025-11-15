#!/bin/bash
# NeuroFlux Health Check Script

SERVICES=("api" "dashboard" "prometheus" "grafana")
PORTS=("8000" "3000" "9090" "3001")

echo "🔍 NeuroFlux Health Check"
echo "========================"

all_healthy=true

for i in "${!SERVICES[@]}"; do
    service=${SERVICES[$i]}
    port=${PORTS[$i]}

    echo -n "Checking $service (port $port)... "

    if curl -f -s --max-time 10 "http://localhost:$port" > /dev/null 2>&1; then
        echo "✅ HEALTHY"
    else
        echo "❌ UNHEALTHY"
        all_healthy=false
    fi
done

echo ""
if [ "$all_healthy" = true ]; then
    echo "🎉 All services are healthy!"
    exit 0
else
    echo "⚠️  Some services are unhealthy. Check logs:"
    echo "   docker-compose logs"
    exit 1
fi