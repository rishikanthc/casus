import marimo

__generated_with = "0.2.3"
app = marimo.App()


@app.cell
def __():
    from casus.encoders import RFF
    import numpy as np
    import jax.numpy as jnp
    import holoviews as holo
    from holoviews import opts

    holo.extension("bokeh")
    return RFF, holo, jnp, np, opts


@app.cell
def __():
    from jax.scipy.stats import norm

    return (norm,)


@app.cell
def __(np):
    data = np.random.normal(size=(2000, 1))
    eval = np.linspace(-5, 5, 1000).reshape(-1, 1)
    return data, eval


@app.cell
def __(RFF, data, jnp):
    def scotts_rule(n: int, d: int):
        return jnp.power(n, -1 / (d + 4))

    bw = scotts_rule(2000, 1) * data.std() * jnp.eye(1)
    encoder = RFF(1, 2048, bw)
    _hv = encoder(data)
    p = _hv.set()
    return bw, encoder, p, scotts_rule


@app.cell
def __(bw, encoder, eval, holo, jnp, norm, p):
    _const = 2000 * bw.squeeze() * (jnp.pi * 2) ** 0.5
    eval_hv = encoder(eval)
    probs = eval_hv.dot(p).T.squeeze() / _const
    gt = norm.pdf(eval).squeeze()

    _p1 = holo.Curve((eval.squeeze(), probs), "x", "p(x)", label="Estimated PDF").opts(
        width=600, height=400
    )
    _p2 = holo.Curve((eval.squeeze(), gt), "x", "p(x)", label="Ground Truth PDF").opts(
        width=600, height=400
    )
    _p1 * _p2
    return eval_hv, gt, probs


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
