import os
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_dataset_metadata(input_path, dataset_name):
    metadata_path = os.path.join(input_path, dataset_name, "metadata.yaml")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def generate_slurm_script(tool_name, tool_config, general_config, dataset_name, output_dir):
    output_path = os.path.join(
        general_config["output_base_path"],
        dataset_name,
        tool_config["tool_name"]
    )

    def format_arg(key, value, joiner=" "):
        if value is None or (isinstance(value, list) and not value):
            return ""
        if isinstance(value, list):
            return f"--{key} {joiner.join(map(str, value))}"
        return f"--{key} {value}"

    extra_args = ""
    extra_args += f" {format_arg('sample', general_config.get('sample'))}"
    extra_args += f" {format_arg('multi_slice', general_config.get('multi_slice'))}"
    extra_args += f" {format_arg('cluster_option', general_config.get('cluster_option'))}"
    extra_args += f" {format_arg('angle_true', general_config.get('angle_true'))}"
    extra_args += f" {format_arg('overlap', general_config.get('overlap'))}"
    extra_args += f" {format_arg('pseudocount', general_config.get('pseudocount'))}"
    extra_args += f" {format_arg('distortion', general_config.get('distortion'))}"
    extra_args += f" {format_arg('subsample', general_config.get('subsample'))}"

    script = f"""#!/bin/bash
#SBATCH --job-name={dataset_name}_{tool_name}
#SBATCH --partition={general_config['partition']}
#SBATCH --nodes=1
#SBATCH --gres=gpu:{tool_config['gpus']}
#SBATCH --cpus-per-task={tool_config['cpus']}
#SBATCH --mem={general_config['memory']}G
#SBATCH --time={general_config['time']}
#SBATCH --output={output_dir}/{dataset_name}_{tool_name}_%j.out

export CUDA_VISIBLE_DEVICES=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | head -n 1)
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

/net/mulan/home/penghuy/anaconda3/envs/{tool_config['env_name']}/bin/python {general_config['benchmarking_py']} \\
  --input_path {general_config['input_path']}/{dataset_name} \\
  --output_path {output_path} \\
  {format_arg('metrics', tool_config.get('metrics'))} \\
  {format_arg('rigid', tool_config.get('rigid'))} \\
  --rep {general_config['rep']} \\
  --tool_name {tool_config['tool_name']} {extra_args}
"""

    script = "\n".join([line for line in script.splitlines() if line.strip()])
    script_path = os.path.join(output_dir, f"{dataset_name}_{tool_name}.slurm")
    with open(script_path, 'w') as f:
        f.write(script)
    return script_path

def submit_job(script_path):
    os.system(f"sbatch {script_path}")

def main():
    config_path = "config.yaml"
    output_dir = "slurm_scripts"
    os.makedirs(output_dir, exist_ok=True)

    config = load_config(config_path)
    general_config = config["general"]

    dataset_names = general_config["dataset_names"]  

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    tools_config = {tool_name: config[tool_name] for tool_name in config if tool_name != "general"}
    selected_tools = general_config.get("selected_tools", [])

    for dataset_name in dataset_names:
        print(f"Processing dataset: {dataset_name}")
        metadata = load_dataset_metadata(general_config["input_path"], dataset_name)
        general_config.update(metadata)

        for tool_name in selected_tools:
            if tool_name in tools_config:
                print(f"Generating and submitting job for tool: {tool_name}, dataset: {dataset_name}")
                script_path = generate_slurm_script(tool_name, tools_config[tool_name], general_config, dataset_name, output_dir)
                submit_job(script_path)
            else:
                print(f"Tool '{tool_name}' not found in the tools configuration. Skipping.")

if __name__ == "__main__":
    main()
    
# export CUDA_VISIBLE_DEVICES=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | head -n 1)
# export CUDA_VISIBLE_DEVICES=1