# code-llama-py
Deploy Code Llama-2-python on Openshift

Set the `HF_TOKEN` environment variable before deploying.

```
oc new-app python~https://github.com/bkoz/code-llama-py --env HF_TOKEN=<my-hf-token>
```