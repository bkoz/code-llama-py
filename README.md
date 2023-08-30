# code-llama-py
Deploy CodeLlama plyground on Openshift using the Hugging Face API.

- Obtain a [Hugging Face user access token](https://huggingface.co/settings/tokens).
- Pass the token into Openshift as an environment variable.

Example
```
oc new-app python~https://github.com/bkoz/code-llama-py --env HF_TOKEN=<my-hf-token>
```

#### References

[EinfachOlder/codellama-playground](https://huggingface.co/spaces/EinfachOlder/codellama-playground/tree/main)
