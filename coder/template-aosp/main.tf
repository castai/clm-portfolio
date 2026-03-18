terraform {
  required_providers {
    coder = {
      source = "coder/coder"
    }
    kubernetes = {
      source = "hashicorp/kubernetes"
    }
  }
}

provider "coder" {}

provider "kubernetes" {
  # When Coder runs inside the cluster, no kubeconfig is needed.
  # The provisioner uses the pod's ServiceAccount.
}

# ---------------------------------------------------------------------------
# Parameters — exposed in the Coder UI when creating a workspace
# ---------------------------------------------------------------------------

variable "namespace" {
  type        = string
  description = "Kubernetes namespace for workspaces"
  default     = "coder"
}

data "coder_parameter" "cpu" {
  name         = "cpu"
  display_name = "CPU cores"
  description  = "Number of CPU cores for the workspace"
  default      = "32"
  icon         = "/icon/memory.svg"
  mutable      = true
  option {
    name  = "16 Cores"
    value = "16"
  }
  option {
    name  = "32 Cores"
    value = "32"
  }
  option {
    name  = "48 Cores"
    value = "48"
  }
  option {
    name  = "64 Cores"
    value = "64"
  }
}

data "coder_parameter" "memory" {
  name         = "memory"
  display_name = "Memory limit (GB)"
  description  = "Memory limit in GB for the workspace"
  default      = "128"
  icon         = "/icon/memory.svg"
  mutable      = true
  option {
    name  = "64 GB"
    value = "64"
  }
  option {
    name  = "96 GB"
    value = "96"
  }
  option {
    name  = "128 GB"
    value = "128"
  }
  option {
    name  = "192 GB"
    value = "192"
  }
  option {
    name  = "256 GB"
    value = "256"
  }
}

data "coder_parameter" "memory_request" {
  name         = "memory_request"
  display_name = "Memory request (GB)"
  description  = "Memory request (guaranteed) in GB"
  default      = "64"
  icon         = "/icon/memory.svg"
  mutable      = true
  option {
    name  = "32 GB"
    value = "32"
  }
  option {
    name  = "64 GB"
    value = "64"
  }
  option {
    name  = "96 GB"
    value = "96"
  }
  option {
    name  = "128 GB"
    value = "128"
  }
}

data "coder_parameter" "home_disk_size" {
  name         = "home_disk_size"
  display_name = "Home disk size (GB)"
  description  = "Size of the persistent home volume (AOSP needs ~400-500GB)"
  default      = "500"
  type         = "number"
  icon         = "/emojis/1f4be.png"
  mutable      = false
  validation {
    min = 100
    max = 1000
  }
}

# ---------------------------------------------------------------------------
# Coder workspace & agent
# ---------------------------------------------------------------------------

data "coder_workspace" "me" {}
data "coder_workspace_owner" "me" {}

resource "coder_agent" "main" {
  os   = "linux"
  arch = "amd64"

  startup_script = <<-EOT
    set -e

    # ── Install AOSP build dependencies ──────────────────────────────────────
    # Reference: https://source.android.com/docs/setup/start
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update -qq
    sudo apt-get install -y -qq --no-install-recommends \
        build-essential bc bison flex git git-lfs curl wget zip unzip \
        python3 python3-pip python-is-python3 \
        openjdk-17-jdk-headless \
        libssl-dev libncurses-dev libelf-dev \
        rsync cpio kmod ccache \
        fontconfig libxml2-utils xsltproc \
        gnupg ca-certificates lsb-release \
        >/dev/null 2>&1

    # ── Install Google's repo tool ───────────────────────────────────────────
    # Reference: https://source.android.com/docs/setup/download#installing-repo
    if [ ! -f /usr/local/bin/repo ]; then
        sudo curl -fsSL -o /usr/local/bin/repo \
            https://storage.googleapis.com/git-repo-downloads/repo
        sudo chmod +x /usr/local/bin/repo
    fi

    # ── Configure git (required by repo) ─────────────────────────────────────
    git config --global user.name "Coder AOSP Builder" 2>/dev/null || true
    git config --global user.email "aosp@coder.local" 2>/dev/null || true
    git config --global color.ui false 2>/dev/null || true

    # ── Write sync-aosp helper script ────────────────────────────────────────
    # Reproduces the customer's exact steps:
    #   1. mkdir AOSP && cd AOSP
    #   2. repo init (partial-clone, no-use-superproject, android-latest-release)
    #   3. repo sync -c -j8
    sudo tee /usr/local/bin/sync-aosp > /dev/null <<'SCRIPT'
    #!/usr/bin/env bash
    set -ex
    AOSP_DIR="$HOME/AOSP"
    mkdir -p "$AOSP_DIR"
    cd "$AOSP_DIR"

    if [ ! -d .repo ]; then
        echo "=== Initializing AOSP repo ==="
        repo init \
            --partial-clone \
            --no-use-superproject \
            -b android-latest-release \
            -u https://android.googlesource.com/platform/manifest
    else
        echo "=== Repo already initialized, skipping init ==="
    fi

    echo "=== Syncing AOSP source (this will take several hours) ==="
    repo sync -c -j8

    echo "=== AOSP source sync complete ==="
    du -sh "$AOSP_DIR"
    SCRIPT
    sudo chmod +x /usr/local/bin/sync-aosp

    # ── Write build-aosp helper script ───────────────────────────────────────
    # Reproduces the customer's exact build steps:
    #   1. source build/envsetup.sh
    #   2. lunch aosp_cf_x86_64_phone-userdebug
    #   3. make -j16
    sudo tee /usr/local/bin/build-aosp > /dev/null <<'SCRIPT'
    #!/usr/bin/env bash
    set -ex
    AOSP_DIR="$HOME/AOSP"

    if [ ! -f "$AOSP_DIR/build/envsetup.sh" ]; then
        echo "ERROR: AOSP source not found. Run 'sync-aosp' first."
        exit 1
    fi

    cd "$AOSP_DIR"

    # AOSP's envsetup.sh uses uninitialized variables, so disable -u
    set +u
    source build/envsetup.sh
    lunch aosp_cf_x86_64_phone trunk_staging userdebug
    set -u

    make -j$(nproc)

    echo "=== AOSP build complete ==="
    SCRIPT
    sudo chmod +x /usr/local/bin/build-aosp

    echo "=== AOSP workspace ready ==="
    echo "  1. Run 'sync-aosp' to download AOSP source (~100-200GB, takes hours)"
    echo "  2. Run 'build-aosp' to build (takes 4-8 hours)"
  EOT

  metadata {
    display_name = "CPU Usage"
    key          = "0_cpu_usage"
    script       = "coder stat cpu"
    interval     = 10
    timeout      = 1
  }

  metadata {
    display_name = "RAM Usage"
    key          = "1_ram_usage"
    script       = "coder stat mem"
    interval     = 10
    timeout      = 1
  }

  metadata {
    display_name = "Home Disk"
    key          = "3_home_disk"
    script       = "coder stat disk --path $${HOME}"
    interval     = 60
    timeout      = 1
  }

  metadata {
    display_name = "CPU Usage (Host)"
    key          = "4_cpu_usage_host"
    script       = "coder stat cpu --host"
    interval     = 10
    timeout      = 1
  }

  metadata {
    display_name = "Memory Usage (Host)"
    key          = "5_mem_usage_host"
    script       = "coder stat mem --host"
    interval     = 10
    timeout      = 1
  }
}

# ---------------------------------------------------------------------------
# Persistent home volume — gp3 for high IOPS during repo sync + build
# ---------------------------------------------------------------------------

resource "kubernetes_persistent_volume_claim_v1" "home" {
  metadata {
    name      = "coder-${data.coder_workspace.me.id}-home"
    namespace = var.namespace
    labels = {
      "app.kubernetes.io/name"     = "coder-pvc"
      "app.kubernetes.io/instance" = "coder-pvc-${data.coder_workspace.me.id}"
      "app.kubernetes.io/part-of"  = "coder"
      "com.coder.resource"         = "true"
      "com.coder.workspace.id"     = data.coder_workspace.me.id
      "com.coder.workspace.name"   = data.coder_workspace.me.name
      "com.coder.user.id"          = data.coder_workspace_owner.me.id
      "com.coder.user.username"    = data.coder_workspace_owner.me.name
    }
  }
  wait_until_bound = false
  spec {
    access_modes       = ["ReadWriteOnce"]
    storage_class_name = "gp3"
    resources {
      requests = {
        storage = "${data.coder_parameter.home_disk_size.value}Gi"
      }
    }
  }
}

# ---------------------------------------------------------------------------
# Workspace deployment — with CAST AI live-migration support
# ---------------------------------------------------------------------------

resource "kubernetes_deployment_v1" "main" {
  count            = data.coder_workspace.me.start_count
  depends_on       = [kubernetes_persistent_volume_claim_v1.home]
  wait_for_rollout = false

  metadata {
    name      = "coder-${data.coder_workspace.me.id}"
    namespace = var.namespace
    labels = {
      "app.kubernetes.io/name"     = "coder-workspace"
      "app.kubernetes.io/instance" = "coder-workspace-${data.coder_workspace.me.id}"
      "app.kubernetes.io/part-of"  = "coder"
      "com.coder.resource"         = "true"
      "com.coder.workspace.id"     = data.coder_workspace.me.id
      "com.coder.workspace.name"   = data.coder_workspace.me.name
      "com.coder.user.id"          = data.coder_workspace_owner.me.id
      "com.coder.user.username"    = data.coder_workspace_owner.me.name
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        "app.kubernetes.io/name" = "coder-workspace"
        "com.coder.workspace.id" = data.coder_workspace.me.id
      }
    }

    strategy {
      type = "Recreate"
    }

    template {
      metadata {
        labels = {
          "app.kubernetes.io/name"     = "coder-workspace"
          "app.kubernetes.io/instance" = "coder-workspace-${data.coder_workspace.me.id}"
          "app.kubernetes.io/part-of"  = "coder"
          "com.coder.resource"         = "true"
          "com.coder.workspace.id"     = data.coder_workspace.me.id
          "com.coder.workspace.name"   = data.coder_workspace.me.name
          "com.coder.user.id"          = data.coder_workspace_owner.me.id
          "com.coder.user.username"    = data.coder_workspace_owner.me.name

          # -- CAST AI live-migration labels --
          "live.cast.ai/migration-enabled" = "true"
        }
      }

      spec {
        security_context {
          run_as_user            = 1000
          fs_group               = 1000
          fs_group_change_policy = "OnRootMismatch"
        }

        # -- CAST AI node placement --
        node_selector = {
          "scheduling.cast.ai/node-template" = "live-migration"
        }

        toleration {
          key      = "scheduling.cast.ai/node-template"
          operator = "Exists"
          effect   = "NoSchedule"
        }

        toleration {
          key      = "scheduling.cast.ai/live-migration"
          operator = "Exists"
          effect   = "NoSchedule"
        }

        toleration {
          key    = "live.cast.ai"
          value  = "true"
          effect = "NoSchedule"
        }

        toleration {
          key                = "node.kubernetes.io/not-ready"
          operator           = "Exists"
          effect             = "NoExecute"
          toleration_seconds = 300
        }

        toleration {
          key                = "node.kubernetes.io/unreachable"
          operator           = "Exists"
          effect             = "NoExecute"
          toleration_seconds = 300
        }

        container {
          name              = "dev"
          image             = "codercom/enterprise-base:ubuntu"
          image_pull_policy = "Always"
          command           = ["sh", "-c", coder_agent.main.init_script]

          security_context {
            run_as_user = "1000"
          }

          env {
            name  = "CODER_AGENT_TOKEN"
            value = coder_agent.main.token
          }

          # Disable MPTCP in the Go runtime — CRIU cannot checkpoint
          # MPTCP sockets (proto 262). Go >= 1.21 enables MPTCP by
          # default when the kernel supports it.
          env {
            name  = "GODEBUG"
            value = "multipathtcp=0"
          }

          # ccache directory for faster rebuilds after migration
          env {
            name  = "CCACHE_DIR"
            value = "/home/coder/.ccache"
          }

          # Use ccache for AOSP builds
          env {
            name  = "USE_CCACHE"
            value = "1"
          }

          resources {
            requests = {
              "cpu"    = data.coder_parameter.cpu.value
              "memory" = "${data.coder_parameter.memory_request.value}Gi"
            }
            limits = {
              "cpu"    = data.coder_parameter.cpu.value
              "memory" = "${data.coder_parameter.memory.value}Gi"
            }
          }

          volume_mount {
            mount_path = "/home/coder"
            name       = "home"
            read_only  = false
          }
        }

        volume {
          name = "home"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim_v1.home.metadata.0.name
            read_only  = false
          }
        }

        affinity {
          pod_anti_affinity {
            preferred_during_scheduling_ignored_during_execution {
              weight = 1
              pod_affinity_term {
                topology_key = "kubernetes.io/hostname"
                label_selector {
                  match_expressions {
                    key      = "app.kubernetes.io/name"
                    operator = "In"
                    values   = ["coder-workspace"]
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
