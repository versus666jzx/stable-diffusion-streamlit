# Latest Stable-Diffusion model 
![Stable-Diffusion app](/images/web_page.png)

### Manual

In this post we'll use model version v1-4, so before use you'll need to visit [this card](https://huggingface.co/CompVis/stable-diffusion-v1-4), read the license and tick the checkbox if you agree.

1. Make sure you're logged in Huggingface

Example for terminal:

```bash
huggingface-cli login
```

Example for notebook:

```python
from huggingface_hub import notebook_login

notebook_login()
```

2. Install requirements

```bash
pip install -r requirements.txt
```

3. Run app.py

```bash
streamlit run path_to_app.py
```

---

| based on: | Streamlit  |
|-----------|------------|
| update:   | 23.08.2022 |

