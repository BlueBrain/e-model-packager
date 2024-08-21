"""Represents common tasks that can be reused in multiple workflows."""

import os
import subprocess

import luigi
from luigi_tools.task import RemoveCorruptedOutputMixin

from e_model_packager.config_decorator import ConfigDecorator


class SmartTask(RemoveCorruptedOutputMixin, luigi.Task):
    """A smarter task that automatically removes output of failed tasks.

    This is to ensure that no corrupted or incomplete output
    gets generated if the Task fails unexpectedly.
    This is the default behaviour in other wfms such as snakemake.
    """

    RemoveCorruptedOutputMixin.clean_failed = luigi.BoolParameter(
        significant=False,
        default=True,
        description="Trigger to remove the outputs of the failed tasks.",
        parsing=luigi.BoolParameter.EXPLICIT_PARSING,
    )


class CreateSystemLog(SmartTask):
    """Task to log the modules and python packages used in the execution."""

    workflow_config = ConfigDecorator(luigi.configuration.get_config())

    def output(self):
        """A log file to be written."""
        workflow_output_dir = self.workflow_config.get("paths", "output")
        return luigi.LocalTarget(os.path.join(workflow_output_dir, "system-state.log"))

    def run(self):
        """Writes down the loaded modules, pip packages and python version."""
        output_dir = self.workflow_config.get("paths", "output")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        module_list = subprocess.run(
            ["modulecmd", "bash", "list"], capture_output=True, check=True
        )
        py_version = subprocess.run(
            ["python", "--version"], capture_output=True, check=True
        )
        pip_list = subprocess.run(["pip", "list"], capture_output=True, check=True)

        modules = " ".join(
            [x.decode("utf-8") for x in [module_list.stdout, module_list.stderr]]
        )
        python_ver, pip = [x.stdout.decode("utf-8") for x in [py_version, pip_list]]

        with self.output().open("w") as outfile:
            outfile.write(f"{modules}\n{python_ver}\n{pip}")
