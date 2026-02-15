#!/bin/bash
# ============================================================================
# deploy.sh — 一键部署推理服务（Docker / K8s 版）
# ============================================================================
#
# 这个脚本提供了多种部署方式的一键启动：
# 1. docker   : 使用 Docker Compose 部署（包含监控栈）
# 2. k8s      : 使用 kubectl 部署到 Kubernetes 集群
# 3. helm     : 使用 Helm Chart 部署到 Kubernetes 集群
#
# 使用方法：
#   bash scripts/deploy.sh docker     # Docker 部署
#   bash scripts/deploy.sh k8s        # Kubernetes 部署
#   bash scripts/deploy.sh helm       # Helm 部署
#   bash scripts/deploy.sh docker down  # 停止 Docker 部署
# ============================================================================

set -euo pipefail

# ---- 颜色定义 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ---- 项目根目录 ----
# dirname $0: 获取脚本所在目录
# cd .. : 回到项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- 使用说明 ----
usage() {
    echo "用法: bash scripts/deploy.sh <部署方式> [操作]"
    echo ""
    echo "部署方式:"
    echo "  docker     Docker Compose 部署（推荐入门使用）"
    echo "  k8s        Kubernetes 部署（使用 kubectl）"
    echo "  helm       Helm Chart 部署（推荐生产使用）"
    echo ""
    echo "操作（可选）:"
    echo "  up         启动服务（默认）"
    echo "  down       停止服务"
    echo "  status     查看服务状态"
    echo "  logs       查看服务日志"
    echo ""
    echo "示例:"
    echo "  bash scripts/deploy.sh docker           # Docker 启动"
    echo "  bash scripts/deploy.sh docker down       # Docker 停止"
    echo "  bash scripts/deploy.sh docker logs       # Docker 日志"
    echo "  bash scripts/deploy.sh k8s               # K8s 部署"
    echo "  bash scripts/deploy.sh helm              # Helm 部署"
}

# ---- Docker 部署 ----
deploy_docker() {
    local action="${1:-up}"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Docker Compose 部署${NC}"
    echo -e "${BLUE}========================================${NC}"

    # 检查 Docker 是否安装
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}[错误] Docker 未安装${NC}"
        echo "请参考: https://docs.docker.com/get-docker/"
        exit 1
    fi

    local compose_file="$PROJECT_ROOT/deploy/docker/docker-compose.yml"

    case $action in
        up)
            echo -e "${YELLOW}[1/2] 构建 Docker 镜像...${NC}"
            docker build -t qwen3-fc-inference \
                -f "$PROJECT_ROOT/deploy/docker/Dockerfile.inference" \
                "$PROJECT_ROOT"

            echo -e "${YELLOW}[2/2] 启动服务栈...${NC}"
            docker compose -f "$compose_file" up -d

            echo ""
            echo -e "${GREEN}[✓] 服务已启动！${NC}"
            echo -e "  vLLM API   : http://localhost:8000"
            echo -e "  Prometheus : http://localhost:9090"
            echo -e "  Grafana    : http://localhost:3000 (admin/admin)"
            ;;
        down)
            echo -e "${YELLOW}停止服务...${NC}"
            docker compose -f "$compose_file" down
            echo -e "${GREEN}[✓] 服务已停止${NC}"
            ;;
        status)
            docker compose -f "$compose_file" ps
            ;;
        logs)
            docker compose -f "$compose_file" logs -f --tail=100
            ;;
        *)
            echo -e "${RED}[错误] 未知操作: $action${NC}"
            usage
            exit 1
            ;;
    esac
}

# ---- Kubernetes 部署 ----
deploy_k8s() {
    local action="${1:-up}"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Kubernetes 部署${NC}"
    echo -e "${BLUE}========================================${NC}"

    # 检查 kubectl 是否安装
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}[错误] kubectl 未安装${NC}"
        echo "请参考: https://kubernetes.io/docs/tasks/tools/"
        exit 1
    fi

    # 检查集群连接
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}[错误] 无法连接到 Kubernetes 集群${NC}"
        echo "请检查 kubeconfig 配置"
        exit 1
    fi

    case $action in
        up)
            echo -e "${YELLOW}[1/2] 创建 Deployment...${NC}"
            kubectl apply -f "$PROJECT_ROOT/deploy/k8s/deployment.yaml"

            echo -e "${YELLOW}[2/2] 创建 Service...${NC}"
            kubectl apply -f "$PROJECT_ROOT/deploy/k8s/service.yaml"

            echo ""
            echo -e "${GREEN}[✓] K8s 资源已创建！${NC}"
            echo -e "${YELLOW}等待 Pod 启动中...${NC}"
            kubectl rollout status deployment/qwen3-fc-inference --timeout=300s || true

            echo ""
            echo "查看 Pod 状态:"
            kubectl get pods -l app=qwen3-fc
            echo ""
            echo "查看 Service:"
            kubectl get svc qwen3-fc-service
            echo ""
            echo -e "${YELLOW}提示：使用 port-forward 访问服务:${NC}"
            echo "  kubectl port-forward svc/qwen3-fc-service 8000:8000"
            ;;
        down)
            echo -e "${YELLOW}删除 K8s 资源...${NC}"
            kubectl delete -f "$PROJECT_ROOT/deploy/k8s/service.yaml" --ignore-not-found
            kubectl delete -f "$PROJECT_ROOT/deploy/k8s/deployment.yaml" --ignore-not-found
            echo -e "${GREEN}[✓] K8s 资源已删除${NC}"
            ;;
        status)
            echo "Pods:"
            kubectl get pods -l app=qwen3-fc
            echo ""
            echo "Services:"
            kubectl get svc qwen3-fc-service
            ;;
        logs)
            kubectl logs -f -l app=qwen3-fc --tail=100
            ;;
        *)
            echo -e "${RED}[错误] 未知操作: $action${NC}"
            usage
            exit 1
            ;;
    esac
}

# ---- Helm 部署 ----
deploy_helm() {
    local action="${1:-up}"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Helm Chart 部署${NC}"
    echo -e "${BLUE}========================================${NC}"

    # 检查 helm 是否安装
    if ! command -v helm &> /dev/null; then
        echo -e "${RED}[错误] Helm 未安装${NC}"
        echo "请参考: https://helm.sh/docs/intro/install/"
        exit 1
    fi

    local chart_path="$PROJECT_ROOT/deploy/helm"
    local release_name="qwen3-fc"

    case $action in
        up)
            echo -e "${YELLOW}安装/升级 Helm Chart...${NC}"
            # helm upgrade --install: 如果不存在就安装，存在就升级
            helm upgrade --install "$release_name" "$chart_path" \
                --wait \
                --timeout 10m

            echo ""
            echo -e "${GREEN}[✓] Helm 部署完成！${NC}"
            helm status "$release_name"
            ;;
        down)
            echo -e "${YELLOW}卸载 Helm Release...${NC}"
            helm uninstall "$release_name" || true
            echo -e "${GREEN}[✓] Helm Release 已卸载${NC}"
            ;;
        status)
            helm status "$release_name" 2>/dev/null || echo "Release 不存在"
            ;;
        logs)
            kubectl logs -f -l app.kubernetes.io/instance="$release_name" --tail=100
            ;;
        *)
            echo -e "${RED}[错误] 未知操作: $action${NC}"
            usage
            exit 1
            ;;
    esac
}

# ---- 主入口 ----
if [ $# -lt 1 ]; then
    usage
    exit 1
fi

DEPLOY_METHOD="$1"
ACTION="${2:-up}"

case $DEPLOY_METHOD in
    docker)
        deploy_docker "$ACTION"
        ;;
    k8s|kubernetes)
        deploy_k8s "$ACTION"
        ;;
    helm)
        deploy_helm "$ACTION"
        ;;
    --help|-h)
        usage
        ;;
    *)
        echo -e "${RED}[错误] 未知部署方式: $DEPLOY_METHOD${NC}"
        usage
        exit 1
        ;;
esac
