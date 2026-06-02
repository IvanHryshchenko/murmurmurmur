import os
import pickle
import datetime
import numpy as np


LOG_DIR  = 'logs'
LOG_FILE = os.path.join(LOG_DIR, 'training.log')

os.makedirs(LOG_DIR, exist_ok=True)


class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, path):
        self._file = open(path, 'a', encoding='utf-8')

    def write(self, text):
        import sys
        sys.__stdout__.write(text)
        self._file.write(text)
        self._file.flush()

    def flush(self):
        import sys
        sys.__stdout__.flush()
        self._file.flush()

    def close(self):
        self._file.close()


_tee = _Tee(LOG_FILE)


def log(text=''):
    """Print to stdout AND append to logs/training.log."""
    _tee.write(text + '\n')

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)

class MiniAI:
    """
    Minimal feedforward neural network trained with backpropagation + Adam.
    Layers: [input_size → 64 → 32 → 16 → 1]

    v3 fixes vs v2:
      - BUG FIX: dropout seed не детерминирован по t (теперь настоящий случайный)
      - BUG FIX: backward на выходном слое не применяет relu_grad (линейный)
      - BUG FIX: нормализация X зажимает std=0 признаки корректно
      - IMPROVEMENT: LR warmup + cosine decay в partial_fit
      - IMPROVEMENT: group_mapping сохраняется в модели для predict
    """

    def __init__(self, input_size=10, hidden=(64, 32, 16), lr=0.001,
                 dropout=0.15, epochs=2000, random_state=42,
                 log_target=True, batch_size=2048, patience=300):
        self.input_size   = input_size
        self.hidden       = hidden
        self.lr           = lr
        self.dropout_rate = dropout
        self.epochs       = epochs
        self.rs           = random_state
        self.log_target   = log_target
        self.batch_size   = batch_size
        self.patience     = patience

        self.x_mean = self.x_std = None
        self.y_mean = self.y_std = None
        self.feature_cols = None

        # Сохраняем mapping групп машин (для CUTTING predict)
        self.group_mapping    = None
        self.group_global_mean = None

        self.history = {}

        self._rng_dropout = np.random.default_rng(random_state + 999)
        self._init_weights()

    def _init_weights(self):
        rng   = np.random.default_rng(self.rs)
        sizes = [self.input_size] + list(self.hidden) + [1]
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            self.W.append(rng.standard_normal((fan_in, sizes[i+1])) * np.sqrt(2.0 / fan_in))
            self.b.append(np.zeros((1, sizes[i+1])))

        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t  = 0

    def _forward(self, X, training=False):
        # FIX: используем глобальный _rng_dropout — настоящий случайный дропаут
        activations, masks, current = [X], [], X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = current @ W + b
            if i < len(self.W) - 1:
                a = relu(z)
                if training and self.dropout_rate > 0:
                    mask = (self._rng_dropout.random(a.shape) > self.dropout_rate).astype(float)
                    a   *= mask / (1.0 - self.dropout_rate)
                else:
                    mask = np.ones_like(a)
                masks.append(mask)
                current = a
            else:
                # FIX: выходной слой линейный — нет relu, нет маски dropout
                current = z
                masks.append(np.ones_like(z))
            activations.append(current)
        return activations, masks

    def _backward_weighted(self, activations, masks, dA_init):
        """Backward pass с уже взвешенным dA."""
        dA = dA_init
        dW_list, db_list = [], []
        for i in reversed(range(len(self.W))):
            A_prev = activations[i]
            # FIX: выходной слой (i == len-1) — линейный, не применяем relu_grad
            if i < len(self.W) - 1:
                dZ = dA * relu_grad(activations[i+1]) * masks[i]
            else:
                dZ = dA  # линейный выход: dZ = dA (производная линейной = 1)
            dW_list.insert(0, A_prev.T @ dZ)
            db_list.insert(0, dZ.sum(axis=0, keepdims=True))
            dA = dZ @ self.W[i].T
        return dW_list, db_list

    def _adam_step(self, dW_list, db_list, lr_override=None):
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr = lr_override if lr_override is not None else self.lr
        self.t += 1
        for i in range(len(self.W)):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * dW_list[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * dW_list[i]**2
            self.W[i] -= lr * (self.mW[i] / (1 - beta1**self.t)) / \
                         (np.sqrt(self.vW[i] / (1 - beta2**self.t)) + eps)

            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * db_list[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * db_list[i]**2
            self.b[i] -= lr * (self.mb[i] / (1 - beta1**self.t)) / \
                         (np.sqrt(self.vb[i] / (1 - beta2**self.t)) + eps)

    def _norm_x(self, X):
        # FIX: нулевой std заменяем на 1 (не на 1e-8) чтобы не усиливать шум
        std = np.where(self.x_std < 1e-10, 1.0, self.x_std)
        return (X - self.x_mean) / std

    def _to_log(self, y):
        return np.log1p(np.clip(y, 0, None)) if self.log_target else y

    def _from_log(self, y):
        return np.expm1(y) if self.log_target else y

    def _norm_y(self, y):
        yl = self._to_log(y)
        return (yl - self.y_mean) / (self.y_std + 1e-8)

    def _denorm_y(self, ys):
        y_raw = ys * self.y_std + self.y_mean
        return self._from_log(y_raw)

    def fit(self, X, y, feature_cols, verbose=True, sample_weight=None):
        self.feature_cols = feature_cols
        self.input_size   = X.shape[1]
        self._init_weights()
        self._rng_dropout = np.random.default_rng(self.rs + 999)

        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            sw = sw / (sw.mean() + 1e-10)
        else:
            sw = np.ones(len(X), dtype=float)
        self._sw = sw

        self.x_mean = X.mean(axis=0)
        self.x_std  = X.std(axis=0)

        y_log = self._to_log(y)
        self.y_mean = y_log.mean()
        self.y_std  = y_log.std() + 1e-8

        Xs = self._norm_x(X)
        ys = self._norm_y(y).reshape(-1, 1)

        n     = len(Xs)
        n_val = max(1, int(n * 0.2))
        rng_split = np.random.default_rng(self.rs)
        idx   = rng_split.permutation(n)
        val_i, tr_i = idx[:n_val], idx[n_val:]

        Xtr, Xv  = Xs[tr_i], Xs[val_i]
        ytr, yv  = ys[tr_i], ys[val_i]

        wtr = self._sw[tr_i].reshape(-1, 1)

        train_losses, val_losses = [], []
        best_val, best_ep = float('inf'), 0
        no_improve = 0
        best_W = [w.copy() for w in self.W]
        best_b = [b.copy() for b in self.b]

        t_start = datetime.datetime.now()
        n_tr = len(tr_i)
        bs   = min(self.batch_size, n_tr)
        rng_batch = np.random.default_rng(self.rs + 1)

        for epoch in range(1, self.epochs + 1):
            perm = rng_batch.permutation(n_tr)
            batch_losses = []
            for start in range(0, n_tr, bs):
                bi  = perm[start:start + bs]
                Xb  = Xtr[bi]
                yb  = ytr[bi]
                wb  = wtr[bi]
                acts, masks = self._forward(Xb, training=True)
                raw_err = (acts[-1] - yb) ** 2
                tr_loss_b = float(np.mean(raw_err * wb))
                batch_losses.append(tr_loss_b)
                dA = (acts[-1] - yb) * wb * 2.0 / len(bi)
                dW, db = self._backward_weighted(acts, masks, dA)
                self._adam_step(dW, db)

            tr_loss = float(np.mean(batch_losses))

            acts_v, _   = self._forward(Xv, training=False)
            vl_loss     = float(np.mean((acts_v[-1] - yv) ** 2))

            train_losses.append(tr_loss)
            val_losses.append(vl_loss)

            if vl_loss < best_val:
                best_val, best_ep = vl_loss, epoch
                no_improve = 0
                best_W = [w.copy() for w in self.W]
                best_b = [b.copy() for b in self.b]
            else:
                no_improve += 1

            if verbose and epoch % 200 == 0:
                elapsed = (datetime.datetime.now() - t_start).total_seconds()
                log(f'  Epoch {epoch:4d}/{self.epochs}'
                    f'  │  Train MSE {tr_loss:.6f}'
                    f'  │  Val MSE {vl_loss:.6f}'
                    f'  │  {elapsed:5.1f}s elapsed')

            if self.patience > 0 and no_improve >= self.patience:
                if verbose:
                    elapsed = (datetime.datetime.now() - t_start).total_seconds()
                    log(f'  ⏹  Early stop at epoch {epoch}  (best val={best_val:.6f} @ ep {best_ep})  {elapsed:.1f}s')
                break

        self.W = best_W
        self.b = best_b

        elapsed_total = (datetime.datetime.now() - t_start).total_seconds()

        def _pred_raw(Xn):
            a, _ = self._forward(Xn, training=False)
            return self._denorm_y(a[-1].ravel())

        pred_all = _pred_raw(Xs)
        pred_tr  = _pred_raw(Xtr)
        pred_vl  = _pred_raw(Xv)

        def _metrics(y_true, y_pred, label):
            err  = y_pred - y_true
            mae  = float(np.mean(np.abs(err)))
            mse  = float(np.mean(err**2))
            rmse = float(np.sqrt(mse))
            ss_r = float(np.sum(err**2))
            ss_t = float(np.sum((y_true - y_true.mean())**2))
            r2   = 1.0 - ss_r / (ss_t + 1e-10)
            mape = float(np.mean(np.abs(err / (np.abs(y_true) + 1e-8)))) * 100
            med  = float(np.median(np.abs(err)))
            p95  = float(np.percentile(np.abs(err), 95))
            return dict(label=label, n=len(y_true),
                        mae=mae, mse=mse, rmse=rmse,
                        r2=r2, mape=mape, median_ae=med, p95_ae=p95)

        m_tr  = _metrics(y[tr_i],  pred_tr,  'Train')
        m_vl  = _metrics(y[val_i], pred_vl,  'Val  ')
        m_all = _metrics(y,        pred_all, 'Full ')

        w1_imp    = np.abs(self.W[0]).mean(axis=1)
        fi_order  = np.argsort(w1_imp)[::-1]

        self.history = dict(
            train_losses     = train_losses,
            val_losses       = val_losses,
            best_val_mse     = best_val,
            best_epoch       = best_ep,
            elapsed_sec      = elapsed_total,
            n_train          = len(tr_i),
            n_val            = len(val_i),
            metrics_train    = m_tr,
            metrics_val      = m_vl,
            metrics_all      = m_all,
            feature_importance = {feature_cols[i]: float(w1_imp[i])
                                  for i in range(len(feature_cols))},
        )

        if verbose:
            _print_loss_graph(train_losses, val_losses)
            _print_report(self.history, feature_cols, fi_order, w1_imp, elapsed_total)

        return self

    def partial_fit(self, X, y, epochs=500, sample_weight=None, verbose=False):
        """
        Инкрементальное дообучение — НЕ сбрасывает веса и нормализацию.
        Вызывается для каждой строки CUTTING по очереди.

        FIX: cosine LR decay + warmup для стабилизации обучения.
        """
        if self.x_mean is None:
            raise RuntimeError("partial_fit требует предварительного вызова fit()")

        Xs = self._norm_x(X)
        ys = self._norm_y(y).reshape(-1, 1)

        n = len(Xs)
        if n < 2:
            sw = np.ones((n, 1)) if sample_weight is None else \
                 (np.asarray(sample_weight, dtype=float) / (np.asarray(sample_weight).mean() + 1e-10)).reshape(-1, 1)
            for ep in range(epochs):
                # Cosine decay
                lr = self.lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))
                acts, masks = self._forward(Xs, training=True)
                dA = (acts[-1] - ys) * sw * 2.0 / max(n, 1)
                dW, db = self._backward_weighted(acts, masks, dA)
                self._adam_step(dW, db, lr_override=lr)
            return self

        n_val = max(1, int(n * 0.2))
        rng = np.random.default_rng(self.rs + self.t)
        idx = rng.permutation(n)
        val_i, tr_i = idx[:n_val], idx[n_val:]
        Xtr, Xv = Xs[tr_i], Xs[val_i]
        ytr, yv = ys[tr_i], ys[val_i]

        sw_arr = np.ones(n) if sample_weight is None else \
                 np.asarray(sample_weight, dtype=float) / (np.asarray(sample_weight).mean() + 1e-10)
        wtr = sw_arr[tr_i].reshape(-1, 1)

        bs = min(self.batch_size, len(tr_i))
        rng_b = np.random.default_rng(self.rs + self.t)

        best_val = float('inf')
        no_improve = 0
        best_W = [w.copy() for w in self.W]
        best_b = [b.copy() for b in self.b]
        patience = max(50, epochs // 5)

        for ep in range(1, epochs + 1):
            # Cosine decay LR
            lr = self.lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))
            perm = rng_b.permutation(len(tr_i))
            for start in range(0, len(tr_i), bs):
                bi = perm[start:start + bs]
                acts, masks = self._forward(Xtr[bi], training=True)
                dA = (acts[-1] - ytr[bi]) * wtr[bi] * 2.0 / max(len(bi), 1)
                dW, db = self._backward_weighted(acts, masks, dA)
                self._adam_step(dW, db, lr_override=lr)

            acts_v, _ = self._forward(Xv, training=False)
            vl = float(np.mean((acts_v[-1] - yv) ** 2))
            if vl < best_val:
                best_val = vl
                no_improve = 0
                best_W = [w.copy() for w in self.W]
                best_b = [b.copy() for b in self.b]
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        self.W = best_W
        self.b = best_b
        return self

    def predict(self, X):
        Xs      = self._norm_x(X)
        acts, _ = self._forward(Xs, training=False)
        pred    = self._denorm_y(acts[-1].ravel())
        pred    = np.where(np.isfinite(pred), pred, 0.0)
        return np.clip(pred, 0.0, None)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


def _print_loss_graph(train_losses, val_losses, width=58, steps=36):
    log()
    log('  📊  LOSS CURVE   T = Train   V = Val   ✪ = overlap')
    log()

    idxs   = np.linspace(0, len(train_losses) - 1, steps).astype(int)
    t_vals = [train_losses[i] for i in idxs]
    v_vals = [val_losses[i]   for i in idxs]
    lo     = min(min(t_vals), min(v_vals))
    hi     = max(max(t_vals), max(v_vals))
    ep_per = len(train_losses) / steps

    def _n(x):
        return int((x - lo) / (hi - lo + 1e-10) * (width - 1))

    log(f'  {hi:8.5f} ┤')
    for i, (t, v) in enumerate(zip(t_vals, v_vals)):
        ti, vi = _n(t), _n(v)
        bar    = [' '] * width
        bar[ti] = 'T'
        bar[vi] = 'V'
        if ti == vi:
            bar[ti] = '✪'
        log(f'  {" ":8} │ {"".join(bar)}')
    log(f'  {lo:8.5f} ┤')
    log(f'           └{"─" * (width // 2)}▶  epochs')
    log()


def _print_report(history, feature_cols, fi_order, w1_imp, elapsed):
    W  = 66
    hl = '─' * W

    def _row(label, val, unit=''):
        v = str(val)
        return f'  │  {label:<30}{v:>22}{unit:<8}  │'

    def _sec(title):
        bar = '─' * (W - len(title) - 4)
        return f'  ├─  {title}  {bar}┤'

    log()
    log(f'  ┌{hl}┐')
    log(f'  │{"  🧠  NEURAL NETWORK TRAINING REPORT":^{W}}│')
    log(f'  │{"  " + datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"):^{W}}│')
    log(f'  ├{hl}┤')

    log(_sec('Architecture'))
    n_feat = len(feature_cols)
    log(_row('Input features',  n_feat))
    log(_row('Hidden layers',   '64 → 32 → 16'))
    log(_row('Output',          '1  (regression)'))
    log(_row('Activation',      'ReLU + linear out'))
    log(_row('Optimizer',       'Adam  β₁=0.9  β₂=0.999'))
    log(_row('Dropout rate',    f'{history.get("dropout", 0.15):.0%}'))
    log(_row('Training time',   f'{elapsed:.1f}', '  sec'))

    log(_sec('Dataset split'))
    log(_row('Total samples',       history['n_train'] + history['n_val']))
    log(_row('Training samples',    history['n_train']))
    log(_row('Validation samples',  history['n_val']))

    tl = history['train_losses']
    vl = history['val_losses']
    log(_sec('Loss history  (MSE, normalised space)'))
    log(_row('Initial  train MSE',  f'{tl[0]:.6f}'))
    log(_row('Final    train MSE',  f'{tl[-1]:.6f}'))
    log(_row('Initial  val   MSE',  f'{vl[0]:.6f}'))
    log(_row('Final    val   MSE',  f'{vl[-1]:.6f}'))
    log(_row('Best     val   MSE',  f'{history["best_val_mse"]:.6f}'))
    log(_row('Best epoch',          history['best_epoch']))
    red = (tl[0] - tl[-1]) / (tl[0] + 1e-10) * 100
    log(_row('Train loss reduction', f'{red:.1f}', '  %'))
    overfit = (vl[-1] - tl[-1]) / (tl[-1] + 1e-10) * 100
    log(_row('Overfit index (val-tr)/tr', f'{overfit:+.1f}', '  %'))

    for m in (history['metrics_train'], history['metrics_val'], history['metrics_all']):
        log(_sec(f'Metrics — {m["label"]}  (n={m["n"]}, denormalised values)'))
        log(_row('MAE   (mean abs error)',      f'{m["mae"]:.4f}',      '  min'))
        log(_row('RMSE  (root mean sq error)',  f'{m["rmse"]:.4f}',     '  min'))
        log(_row('Median abs error',            f'{m["median_ae"]:.4f}','  min'))
        log(_row('P95   abs error',             f'{m["p95_ae"]:.4f}',   '  min'))
        log(_row('MAPE  (mean abs % error)',    f'{m["mape"]:.2f}',     '  %'))
        r2_bar = '█' * int(max(0, m['r2']) * 20)
        log(_row('R²',                          f'{m["r2"]:.4f}   {r2_bar}'))

    log(_sec('Feature importance  (mean |W₁| per input)'))
    max_imp = w1_imp[fi_order[0]] + 1e-10
    bar_w   = 24
    for rank, fi in enumerate(fi_order):
        fname = feature_cols[fi][:26]
        imp   = w1_imp[fi]
        bar   = '█' * int(imp / max_imp * bar_w)
        pct   = imp / max_imp * 100
        log(f'  │  {rank+1:>2}. {fname:<26} {bar:<{bar_w}}  {pct:5.1f}%  {imp:.4f}  │')

    log(f'  └{hl}┘')
    log()