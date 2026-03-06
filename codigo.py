import numpy as np
import matplotlib.pyplot as plt

# =========================================================================================
# SIMULACIÓN OFDM - Código principal
# - Objetivo: simular tramas OFDM con preámbulo STF-like (sin CP en preámbulo), detector STF,
#   estimador coarse de CFO y extracción/ecualización/demap de símbolos.
# - Notación: variables en español usando PascalCase para globals, funciones en CamelCase.
# - Comentarios: explicativos en todo el archivo (tal como pediste).
# =========================================================================================

# ========================= PARÁMETROS GLOBALES (en español, PascalCase) =========================
K = 64                           # cantidad de subportadoras (constante del sistema)
N = 10                           # número inicial de símbolos OFDM (se sobrescribe por input)
BitsPorSimbolo = 2               # QPSK -> 2 bits por portadora
np.set_printoptions(threshold=10000, linewidth=200)

# ------------------ PARÁMETRO DE PRUEBA: CFO real inyectado en Tx (ahora GLOBAL) ----------
# DF_TRUE: offset de frecuencia aplicado en el transmisor para pruebas (en cycles/sample).
DF_TRUE = 0.0

# Respuesta al impulso del canal (por defecto unitario). Cambiá aquí para probar canales dispersivos.
RespuestaImpulsoCanal = np.array([1.0], dtype=complex)

# CP fijo que se usará en toda la simulación. Debe cumplir 0 <= ValorCP < K
ValorCP = 3

# Número de tramas solicitadas por iteración (intento inicial) — el programa ajusta si no divide N.
TramasPorIterSolicitadas = 4

# Símbolos OFDM de silencio entre tramas (fuera de la trama, modela cola de canal)
SimbolosSilencio = 5

# Umbral para detección directa inicial (DetectorSTF produce M(d) y usamos este umbral)
UmbralDeteccionPreambulo = 0.08
# =========================================================================================


# ========================= FUNCIONES PRINCIPALES (con comentarios explicativos) ============
def ConstruirPreambuloSTF(K_local):
    """
    Construye un preámbulo tipo STF en tiempo SIN CP.
    - K_local: número de subportadoras (debe ser par).
    - Se construye una mitad en frecuencia con alternancia [1, -1, 1, -1, ...],
      luego IFFT ortho de la mitad y se repite esa mitad en tiempo dos veces
      para obtener la estructura repetida (STF-like).
    - Devuelve s_stf_time: vector complejo de longitud K_local (símbolo tiempo).
    """
    if K_local % 2 != 0:
        raise ValueError("K debe ser par para construir STF-like con dos mitades iguales.")
    K2 = K_local // 2
    mitad_freq = np.array([1.0 if (i % 2 == 0) else -1.0 for i in range(K2)], dtype=complex)
    a_half = np.fft.ifft(mitad_freq, n=K2, norm='ortho')   # IFFT ortho de la mitad
    s_stf_time = np.tile(a_half, 2)                         # repetir en el tiempo dos veces
    return s_stf_time


def Encoder(K_local, NumeroSimbolos):
    """
    Genera bits aleatorios y mapea a QPSK normalizada (energía por símbolo = 1).
    - num_bits = BitsPorSimbolo * K_local * NumeroSimbolos
    - BitsTx: array 1D de bits (0/1)
    - Xmat: matriz (NumeroSimbolos x K_local) con símbolos complejos QPSK
    - stats: diccionario con media y varianza empíricas de Xmat (útil para debug/registro)
    """
    num_bits = BitsPorSimbolo * K_local * NumeroSimbolos
    BitsTx = np.random.randint(0, 2, size=num_bits)
    bits_reshaped = BitsTx.reshape(NumeroSimbolos, K_local, BitsPorSimbolo)
    bI = bits_reshaped[:, :, 0]
    bQ = bits_reshaped[:, :, 1]
    Xmat = (1 - 2*bI) + 1j * (1 - 2*bQ)   # mapeo duro QPSK: 0->+1, 1->-1
    Xmat = Xmat / np.sqrt(2)              # normalización para energía 1
    mu_before = np.mean(Xmat)
    var_before = np.mean(np.abs(Xmat - mu_before)**2)
    stats = {'media_empirica': mu_before, 'varianza_empirica': var_before}
    return BitsTx, Xmat, stats


def WaveformFormer(Xmat_local, K_local, N_cp_local):
    """
    Convierte símbolos en frecuencia (fila por fila) a tiempo usando IFFT ortho,
    y agrega CP de longitud N_cp_local a cada fila.
    - Si N_cp_local == 0 devuelve solo los símbolos en tiempo (sin CP).
    - Devuelve una matriz (m x (K_local + N_cp_local)) con cada fila = CP | símbolo.
    """
    s_time = np.fft.ifft(Xmat_local, n=K_local, axis=1, norm='ortho')  # IFFT por fila
    if N_cp_local == 0:
        return s_time
    s_with_cp = np.zeros((s_time.shape[0], K_local + N_cp_local), dtype=complex)
    s_with_cp[:, :N_cp_local] = s_time[:, -N_cp_local:]   # copiar último N_cp_local como CP
    s_with_cp[:, N_cp_local:] = s_time
    return s_with_cp


def Canal(x_total, snr_lin, h):
    """
    Modelo de canal lineal + AWGN en flujo continuo:
    - x_total: trama aplanada (1D)
    - h: respuesta al impulso del canal
    - snr_lin: Es/N0 linear
    Devuelve:
      r_total: señal recibida (convolución + ruido)
      y_conv_total: convolución limpia sin ruido (útil para debug)
      PotenciaSenal: potencia empírica de la señal (centrada en media)
      Sigma2: varianza del ruido complejo (por muestra)
    Observación: calculamos potencia sobre la señal centrada (restando media)
    para estimar Sigma2 de forma coherente incluso si hay DC.
    """
    Lh = len(h)
    y_conv_total = np.convolve(x_total, h, mode='full')   # convolución lineal
    mu_emp = np.mean(y_conv_total)
    y_zero_mean = y_conv_total - mu_emp
    PotenciaSenal = np.mean(np.abs(y_zero_mean)**2)
    if PotenciaSenal <= 0:
        PotenciaSenal = 1e-12
    Sigma2 = PotenciaSenal / snr_lin
    noise_re = np.random.randn(*y_conv_total.shape) * np.sqrt(Sigma2 / 2.0)
    noise_im = np.random.randn(*y_conv_total.shape) * np.sqrt(Sigma2 / 2.0)
    noise = noise_re + 1j * noise_im
    r_total = y_conv_total + noise
    return r_total, y_conv_total, PotenciaSenal, Sigma2


def DetectorSTF(r_vec, L, tau_s, step=1):
    """
    Detector tipo STF (métrica clásica):
    - Calcula P(d) = sum_{n=0..L-1} r[d+n] * conj(r[d+n+tau_s])
    - R(d)  = sum |r[d:n]|^2 + sum |r[d+tau_s:n]|^2  (energía local, normalización)
    - M(d)  = |P(d)|^2 / R(d)^2  (métrica de energía-normalizada)
    - Devuelve arrays: ds (posiciones d), P_vec, R_vec, M_vec
    - Nota: protección para denominadores 0 (se usan epsilon).
    """
    Ntot = len(r_vec)
    max_d = Ntot - (tau_s + L)
    if max_d < 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    ds = np.arange(0, max_d + 1, step)
    P_list = []
    R_list = []
    for d in ds:
        a = r_vec[d : d + L]
        b = r_vec[d + tau_s : d + tau_s + L]
        P = np.sum(a * np.conjugate(b))
        # Normalizamos por energía (más robusto frente a variaciones de amplitud)
        R = np.sum(np.abs(a)**2) + np.sum(np.abs(b)**2)
        P_list.append(P)
        R_list.append(R)
    P_vec = np.array(P_list)
    R_vec = np.array(R_list)
    denom = (R_vec**2).copy()
    denom[denom == 0] = 1e-24
    M_vec = (np.abs(P_vec)**2) / denom
    return ds, P_vec, R_vec, M_vec


def find_mesetas(M_vec, d_positions, M_th, min_width):
    """
    Encuentra mesetas (intervalos contiguos) donde M_vec >= M_th.
    - Devuelve lista de tuplas (d_start, d_end, slice(indices)).
    - min_width: ancho mínimo (en índices de M_vec) para aceptar una meseta.
    """
    above = M_vec >= M_th
    mesetas = []
    if above.size == 0:
        return mesetas
    start = None
    for i, val in enumerate(above):
        if val and start is None:
            start = i
        elif not val and start is not None:
            end = i - 1
            if (end - start + 1) >= min_width:
                mesetas.append((d_positions[start], d_positions[end], slice(start, end+1)))
            start = None
    if start is not None:
        end = len(above) - 1
        if (end - start + 1) >= min_width:
            mesetas.append((d_positions[start], d_positions[end], slice(start, end+1)))
    return mesetas


def get_d0(M_vec_slice, d_slice_positions):
    """
    Política auxiliar (la usábamos antes): escoger el primer d dentro de la meseta
    donde M >= 0.5*M_max. NO es la política que vas a imponer si obligás a la mitad,
    pero la dejamos como helper.
    - Retorna (d0, idx_rel) o (None, None) si la slice está vacía.
    """
    if len(M_vec_slice) == 0:
        return None, None
    M_max = np.max(M_vec_slice)
    half = 0.5 * M_max
    for idx_rel, val in enumerate(M_vec_slice):
        if val >= half:
            return int(d_slice_positions[idx_rel]), idx_rel
    mid_idx = len(M_vec_slice) // 2
    return int(d_slice_positions[mid_idx]), mid_idx


def nTupleFormer(y_rows_after_channel, K_local, start_idx_list):
    """
    Quita CP usando los start_idx dados y aplica FFT ortho por fila.
    - y_rows_after_channel: lista de ventanas temporales (cada una puede tener longitud variable)
    - start_idx_list: índices dentro de cada ventana donde empiezan las K muestras útiles (sin CP)
    - Si la ventana no tiene suficientes muestras, se rellena con ceros hasta K.
    - Devuelve Ymat: matriz (n_rows x K_local) con FFT por fila.
    """
    Y_list = []
    for y_row, start in zip(y_rows_after_channel, start_idx_list):
        end = start + K_local
        s = start
        if s < 0:
            s = 0
        if end > y_row.size:
            tmp = np.zeros(K_local, dtype=complex)
            slice_part = y_row[s: y_row.size]
            tmp[:slice_part.size] = slice_part
            y_useful = tmp
        else:
            y_useful = y_row[s:end]
        Y = np.fft.fft(y_useful, n=K_local, norm='ortho')
        Y_list.append(Y)
    if len(Y_list) == 0:
        return np.zeros((0, K_local), dtype=complex)
    Ymat = np.vstack(Y_list)
    return Ymat


def Ecualizador(Ymat, Hhat):
    """
    Ecualizador simple: división por Hhat (estimación del canal por subportadora).
    - Aquí Hhat = 1 (ideal) por simplicidad en la simulación base.
    """
    return Ymat / Hhat


def Demapper(Xhat):
    """
    Demapeo duro QPSK: decisión por signo en I y Q.
    - Entrada Xhat: matriz (n_sym x K)
    - Salida BitsRx: vector 1D (0/1) en el orden de mapeo original.
    """
    if Xhat.size == 0:
        return np.zeros(0, dtype=int)
    bitI = (np.real(Xhat) < 0).astype(int)
    bitQ = (np.imag(Xhat) < 0).astype(int)
    n_sym, K_local = Xhat.shape
    BitsRx = np.stack((bitI, bitQ), axis=2).reshape(n_sym * K_local * 2)
    return BitsRx


def TablaYConstelacion():
    """
    Imprime la tabla de mapeo bits->símbolo y grafica la constelación QPSK.
    (Se mantiene por compatibilidad con tu flujo actual.)
    """
    raw_mapping = {
        '00': 1 + 1j,
        '01': -1 + 1j,
        '11': -1 - 1j,
        '10': 1 - 1j
    }
    norm = np.sqrt(2)
    mapping = {bits: val / norm for bits, val in raw_mapping.items()}
    print("Tabla de mapeo (bits -> símbolo complejo):")
    for bits, sym in mapping.items():
        print(f"  {bits}  ->  {sym.real:+.3f} {sym.imag:+.3f}j")
    plt.figure(figsize=(5,5))
    for bits, sym in mapping.items():
        plt.scatter(np.real(sym), np.imag(sym), s=120)
        plt.text(np.real(sym)+0.03, np.imag(sym)+0.03, bits, fontsize=12)
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.title('Constelación QPSK y tabla de mapeo')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.gca().set_aspect('equal', 'box')
    plt.grid(True)
    plt.show()


# ---------------------- ArmarTrama ----------------------
def ArmarTrama(preambulo_tiempo, s_time_sub):
    """
    Construye la trama aplanada: [preambulo (K) | OFDM1 (K+Ncp) | OFDM2 ...]
    - Devuelve x_total (1D) y data_mask (boolean array indicando OFDM de datos).
    - Nota: no retorna offsets conocidos del TX.
    """
    partes = []
    partes.append(preambulo_tiempo.copy())
    for r in range(s_time_sub.shape[0]):
        partes.append(s_time_sub[r, :])
    x_total = np.concatenate(partes)
    num_datos = s_time_sub.shape[0]
    data_mask = np.array([True] * num_datos)
    return x_total, data_mask


# ========================= Estimador coarse de CFO =========================
def EstimadorCFO(P_slice, R_slice, tau_s):
    """
    Estimador coarse de CFO usando P(d) sobre la meseta:
    - P_slice: valores complejos P(d) dentro de la meseta
    - R_slice: valores reales R(d) (energía) dentro de la meseta
    - tau_s: separación entre repeticiones (muestras)
    Salidas:
      - df_hat: estimación en cycles/sample (float)
      - phi_hat: fase estimada (radianes)
      - P_sum: suma ponderada compleja usada (se devuelve para diagnóstico)
      - ok_flag: True si la estimación supera umbrales de confianza
    Método:
      - P_sum = sum(P_slice * R_slice) / sum(R_slice)  (ponderación por energía local, normalizada)
      - phi_hat = angle(P_sum)
      - df_hat = phi_hat / (2π tau_s)   (nota: convención de signo depende de cómo se inyectó DF_TRUE)
      - ok_flag: magnitud de P_sum mayor que umbral absoluto y relativo a mean(R_slice)
    """
    if P_slice.size == 0:
        return 0.0, 0.0, 0+0j, False
    weights = R_slice.copy()
    sumw = np.sum(weights)
    if sumw == 0:
        weights = np.ones_like(weights)
        sumw = np.sum(weights)
    P_sum = np.sum(P_slice * weights) / sumw
    phi_hat = np.angle(P_sum)
    df_hat = - phi_hat / (2.0 * np.pi * float(tau_s))
    mag = np.abs(P_sum)
    meanR = np.mean(R_slice) if R_slice.size > 0 else 0.0
    ok_flag = (mag > 1e-6) and (mag > (1e-3 * meanR))
    return df_hat, phi_hat, P_sum, ok_flag


# ========================= SIMULACIÓN (función principal) =========================
def RunMonteCarlo(snr_dB_vector, runs, h, N_cp, tramas_por_iter, simbolos_silencio, DF_TRUE):
    """
    Monte Carlo.
    - snr_dB_vector: lista de SNR (Es/N0) en dB
    - runs: iteraciones Monte Carlo por SNR
    - h: respuesta al impulso del canal
    - N_cp: longitud de CP
    - tramas_por_iter: cuantas tramas se transmiten por iteracion
    - simbolos_silencio: cantidad de símbolos de silencio transmitidos fuera de trama
    - DF_TRUE: CFO inyectado en Tx (cycles/sample) - se pasó como argumento por claridad
    Devuelve:
      - ber_means, ber_stds, all_bers_per_snr, ultima_info (dic con info de última iteración)
    """
    ber_means = []
    ber_stds = []
    all_bers_per_snr = []
    ultima_info = None

    NumeroSimbolosTotales = N
    simbolos_por_trama = NumeroSimbolosTotales // tramas_por_iter
    preambulo_tiempo = ConstruirPreambuloSTF(K)

    # parámetros del detector STF (coinciden con preámbulo)
    L = K // 2
    tau_s = K // 2
    min_meseta_width = max(1, N_cp)

    for snr_db in snr_dB_vector:
        snr_lin = 10 ** (snr_db / 10.0)
        print("\n" + "="*72)
        print(f"Simulando SNR = {snr_db:.3f} dB (lin={snr_lin:.3f})  |  DF_TRUE (Tx inj) = {DF_TRUE} cyc/samp")
        print("="*72)
        bers_this_snr = []

        for run in range(runs):
            # ---------------- 1) Generar trama (bits + símbolos en frecuencia) ----------------
            BitsTx, Xmat, stats = Encoder(K, NumeroSimbolosTotales)
            Y_acc_list = []           # acumulador de filas (símbolos) aceptadas por trama
            BitsTxAceptados = []      # acumulador de bits de tramas aceptadas
            PotenciaSenalUltima = None
            Sigma2Ultima = None

            # Recorrer tramas dentro de una iteración
            for t in range(tramas_por_iter):
                # Seleccionar subconjunto de símbolos para esta trama
                i0 = t * simbolos_por_trama
                i1 = i0 + simbolos_por_trama
                X_sub = Xmat[i0:i1, :]

                # Generar waveform en tiempo y agregar CP
                s_time_sub = WaveformFormer(X_sub, K, N_cp)

                # Armar trama concatenando preámbulo (sin CP) y símbolos con CP
                x_total, data_mask = ArmarTrama(preambulo_tiempo, s_time_sub)

                # ---------------- INYECCIÓN DE CFO EN TX (solo para pruebas) ----------------
                if DF_TRUE != 0.0:
                    n_tx = np.arange(len(x_total))
                    x_total = x_total * np.exp(1j * 2.0 * np.pi * DF_TRUE * n_tx)
                    # Para no spamear, imprimimos la inyección solo para run==0 y trama 0
                    if run == 0 and t == 0:
                        print(f"[run 0] Trama {t+1}/{tramas_por_iter}: Inyectado DF_TRUE={DF_TRUE:.6e} cyc/samp en Tx.")

                # ---------------- pasar trama por canal (convolución + AWGN) ----------------
                r_total, y_conv_total, PotenciaSenal, Sigma2 = Canal(x_total, snr_lin, h)
                PotenciaSenalUltima = PotenciaSenal
                Sigma2Ultima = Sigma2

                # ---------------- DETECTOR STF: calcular P(d), R(d), M(d) -------------------
                ds, Pvec, Rvec, Mvec = DetectorSTF(r_total, L=L, tau_s=tau_s, step=1)

                # Variables por trama (se usan después)
                detectado = False
                d0 = None
                df_hat = 0.0
                phi_hat = 0.0
                P_sum_used = 0+0j
                applied_correction = False

                # ------------------- si hay posiciones de M(d) calculadas --------------------
                if Mvec.size > 0:
                    # Encontrar mesetas usando el umbral global UmbralDeteccionPreambulo
                    mesetas = find_mesetas(Mvec, ds, M_th=UmbralDeteccionPreambulo, min_width=min_meseta_width)

                    # Si hay mesetas, elegir la "mejor" por longitud/área (heurística)
                    if len(mesetas) > 0:
                        best = None
                        best_len = -1
                        best_area = -1
                        for (d_start, d_end, sl) in mesetas:
                            length = sl.stop - sl.start
                            area = np.sum(Mvec[sl])
                            if length > best_len or (length == best_len and area > best_area):
                                best = (d_start, d_end, sl)
                                best_len = length
                                best_area = area
                        d_start, d_end, sl = best
                        M_slice = Mvec[sl]
                        ds_slice = ds[sl]

                        # -------- POLÍTICA: usar la MITAD de la meseta como punto de referencia ----
                        d0_mid = int(round((d_start + d_end) / 2.0))

                        # Validación energética local alrededor de d0_mid para evitar falsas detecciones
                        check_len = K + N_cp + len(h)
                        start_check = max(0, d0_mid)
                        end_check = min(len(r_total), d0_mid + check_len)
                        energy_local = np.mean(np.abs(r_total[start_check:end_check])**2)

                        # Impresión por trama solo para run == 0
                        if run == 0:
                            print("-"*60)
                            print(f"[run 0] Trama {t+1}/{tramas_por_iter} -> meseta seleccionada {d_start}..{d_end} (width {sl.stop-sl.start})")
                            print(f"         d0_mid = {d0_mid}, energy_local = {energy_local:.3e}")
                            # también mostramos medias relevantes para diagnóstico conciso
                            print(f"         mean(R) = {np.mean(Rvec[sl]):.3e}, mean(|P|) = {np.mean(np.abs(Pvec[sl])):.3e}")
                        # Si la energía local indica que hay señal:
                        if energy_local > 1e-16:
                            detectado = True
                            d0 = d0_mid  # usamos la mitad como timing reference (política obligada)

                            # -------- ESTIMADOR CFO sobre la meseta (P_slice, R_slice) ------------
                            P_slice = Pvec[sl]
                            R_slice = Rvec[sl]
                            df_hat, phi_hat, P_sum_used, ok_flag = EstimadorCFO(P_slice, R_slice, tau_s)

                            # Impresión resumida del estimador (solo run==0)
                            if run == 0:
                                print(f"         EstCFO: |P_sum|={np.abs(P_sum_used):.3e}, phi_hat={phi_hat:.3f} rad, df_hat={df_hat:.6e} cyc/samp, ok_flag={ok_flag}")

                            # Aplicar corrección multiplicativa referenciada a d0 (si ok_flag True)
                            if ok_flag:
                                n_idx = np.arange(len(r_total))
                                corr = np.exp(-1j * 2.0 * np.pi * df_hat * (n_idx - d0))
                                r_total = r_total * corr
                                applied_correction = True
                                if run == 0:
                                    print(f"         Corrección CFO APLICADA referenciada a d0_mid={d0}.")
                            else:
                                applied_correction = False
                                if run == 0:
                                    print(f"         Corrección CFO NO aplicada (estimador no confiable).")

                # Si no detectamos preámbulo, modelamos silencio y seguimos
                if not detectado:
                    if run == 0:
                        print(f"[run 0] Trama {t+1}/{tramas_por_iter}: PREÁMBULO NO DETECTADO -> trama ignorada.")
                    if simbolos_silencio > 0:
                        silence_concat = np.zeros((K + N_cp) * simbolos_silencio, dtype=complex)
                        _r_sil, _y_conv_sil, PotSil, Sigma2Sil = Canal(silence_concat, snr_lin, h)
                        if run == 0 and t == tramas_por_iter - 1:
                            print(f"         Silencio transmitido despues: PotSil={PotSil:.4e}, sigma2={Sigma2Sil:.3e}")
                    continue

                # ------------------- Extracción de ventanas por símbolo y FFT -----------------
                # Construimos ventanas temporales por cada símbolo (0=preambulo, 1..m=datos)
                m = s_time_sub.shape[0]
                windows = []
                start_idx_list = []
                for idx_sym in range(0, 1 + m):
                    if idx_sym == 0:
                        tx_off = 0
                        len_symbol = K
                    else:
                        tx_off = K + (idx_sym - 1) * (K + N_cp)
                        len_symbol = K + N_cp
                    long_extract = len_symbol + len(h) - 1
                    rx_start = d0 + tx_off
                    rx_end = rx_start + long_extract
                    if rx_start < 0:
                        rx_start = 0
                    if rx_end > len(r_total):
                        rx_end = len(r_total)
                    window = r_total[rx_start:rx_end]
                    windows.append(window)
                    # start_idx indica donde empiezan las K muestras útiles dentro de la ventana
                    if idx_sym == 0:
                        start_idx = (len(h) - 1)
                    else:
                        start_idx = N_cp + (len(h) - 1)
                    start_idx_list.append(int(start_idx))

                # Quitar CP (usando start_idx_list) y FFT por fila
                Y_sub = nTupleFormer(windows, K, start_idx_list)

                # Si hay filas de datos (descartamos fila 0 = preámbulo), apilamos
                if Y_sub.size > 0:
                    if Y_sub.shape[0] > 1:
                        Y_data = Y_sub[1:, :]   # filas de datos solamente (m x K)
                        Y_acc_list.append(Y_data)
                        b0 = i0 * K * BitsPorSimbolo
                        b1 = i1 * K * BitsPorSimbolo
                        BitsTxAceptados.append(BitsTx[b0:b1])
                        if run == 0:
                            print(f"[run 0] Trama {t+1}/{tramas_por_iter}: PREÁMBULO detectado (d0={d0}), datos extraídos, P_signal={PotenciaSenal:.4e}, correction_applied={applied_correction}")
                            print(f"            Y_sub.shape={Y_sub.shape}, Y_data.shape={Y_data.shape}, BitsTxAcept.size={BitsTxAceptados[-1].size}")
                        if simbolos_silencio > 0 and run == 0 and t == tramas_por_iter - 1:
                            silence_concat = np.zeros((K + N_cp) * simbolos_silencio, dtype=complex)
                            _r_sil, _y_conv_sil, PotSil, Sigma2Sil = Canal(silence_concat, snr_lin, h)
                            print(f"            Silencio transmitido despues: PotSil={PotSil:.4e}, sigma2={Sigma2Sil:.3e}")
                    else:
                        if run == 0:
                            print("[run 0] DEBUG: Y_sub solo contiene preámbulo, no hay filas de datos para apilar.")
                else:
                    if run == 0:
                        print(f"[run 0] Trama {t+1}/{tramas_por_iter}: detectada pero Y_sub vacío (ventanas insuficientes).")

            # --------------------- Fin bucle tramas por run ---------------------------------
            # Si no hay símbolos aceptados apilados marcamos BER=1 para esa iteración
            if len(Y_acc_list) == 0:
                ber = 1.0
                bers_this_snr.append(ber)
                ultima_info = {
                    'bits_tx': BitsTx,
                    'bits_rx': np.zeros(0, dtype=int),
                    'Xmat': Xmat,
                    'PotenciaSenal': PotenciaSenalUltima,
                    'Sigma2': Sigma2Ultima,
                    'StartIdx': None,
                    'stats': stats,
                    'snr_db': snr_db,
                    'simbolos_por_trama': simbolos_por_trama,
                    'tramas_por_iter': tramas_por_iter,
                    'Ymat': None,
                    'Xhat': None,
                    'df_hat': None
                }
                continue

            # Apilar todas las filas de símbolos aceptados en una única matriz
            Ymat = np.vstack(Y_acc_list)
            if len(BitsTxAceptados) == 0:
                BitsTxAcept = np.zeros(0, dtype=int)
            else:
                BitsTxAcept = np.concatenate(BitsTxAceptados)

            # Equalización (aquí Hhat=1) y demapeo
            Hhat = np.ones_like(Ymat)
            Xhat = Ecualizador(Ymat, Hhat)
            BitsRx = Demapper(Xhat)

            # Impresión resumen shapes para run==0 (una sola vez por snr)
            if run == 0:
                print(f"[run 0] Post-proc: Ymat.shape={Ymat.shape}, BitsRx.size={BitsRx.size}, BitsTxAcept.size={BitsTxAcept.size}")

            # Comparar bits (truncando a la longitud mínima disponible)
            min_len = min(BitsRx.size, BitsTxAcept.size)
            if min_len == 0:
                ber = 1.0 if BitsRx.size != BitsTxAcept.size else 0.0
            else:
                errores = np.sum(BitsRx[:min_len] != BitsTxAcept[:min_len])
                ber = errores / min_len

            bers_this_snr.append(ber)

            # Guardar info de la última iteración para inspección posterior
            ultima_info = {
                'bits_tx': BitsTx,
                'bits_rx': BitsRx,
                'Xmat': Xmat,
                'PotenciaSenal': PotenciaSenalUltima,
                'Sigma2': Sigma2Ultima,
                'StartIdx': None,
                'stats': stats,
                'snr_db': snr_db,
                'simbolos_por_trama': simbolos_por_trama,
                'tramas_por_iter': tramas_por_iter,
                'Ymat': Ymat,
                'Xhat': Xhat,
                'df_hat': df_hat,
                'phi_hat': phi_hat,
                'P_sum_used': P_sum_used
            }

        # Fin runs por SNR: calcular medias / std y mostrar resumen
        bers_arr = np.array(bers_this_snr)
        ber_means.append(np.mean(bers_arr))
        ber_stds.append(np.std(bers_arr))
        all_bers_per_snr.append(bers_arr)
        print(f"Resumen SNR {snr_db:.3f} dB -> BER_mean = {ber_means[-1]:.3e}, BER_std = {ber_stds[-1]:.3e}")

    print()
    return np.array(ber_means), np.array(ber_stds), all_bers_per_snr, ultima_info


# ========================= EJECUCIÓN PRINCIPAL ============================================
if __name__ == "__main__":
    # Mensaje inicial: aclaramos que DF_TRUE es global y dónde modificarlo
    print("Ejecutando simulación OFDM (canal + AWGN) con preámbulo STF-like (sin CP en preámbulo).")

    # Pedimos total de bits (manteniendo la interacción previa)
    default_bits = BitsPorSimbolo * K * N
    while True:
        try:
            user_input = input("Ingrese número total de bits a transmitir (múltiplo de 128, Enter para por defecto): ")
            if user_input.strip() == "":
                TotalBits = default_bits
                print(f"Usando valor por defecto: {TotalBits} bits.")
                break
            TotalBits = int(user_input)
            if TotalBits <= 0:
                print("Debe ingresar un entero positivo distinto de 0. Intente nuevamente.")
                continue
            if TotalBits % 128 != 0:
                print("El número debe ser múltiplo de 128. Intente nuevamente.")
                continue
            break
        except ValueError:
            print("Entrada inválida. Ingrese un entero múltiplo de 128 o presione Enter para usar el valor por defecto.")

    # Preparamos parámetros de canal y CP
    h = RespuestaImpulsoCanal.copy()
    Lh = len(h)

    N_CP = int(ValorCP)
    if N_CP < 0:
        raise ValueError("ValorCP debe ser >= 0")
    if N_CP >= K:
        print("ValorCP >= K, ajustando a K-1.")
        N_CP = K - 1

    BitsPorOFDM = BitsPorSimbolo * K
    total_bit = TotalBits
    N = total_bit // BitsPorOFDM   # recalculamos N según bits solicitados

    # Ajuste de tramas por iteración si el solicitado no divide N
    requested = int(TramasPorIterSolicitadas)
    if requested <= 0:
        tramas_por_iter = 1
    else:
        if N % requested == 0:
            tramas_por_iter = requested
        else:
            divisores = [d for d in range(min(requested, N), 0, -1) if N % d == 0]
            tramas_por_iter = divisores[0] if len(divisores) > 0 else 1
            if tramas_por_iter != requested:
                print(f"TramasPorIterSolicitadas={requested} no divide N={N}. Se ajusta a tramas_por_iter={tramas_por_iter}.")

    simbolos_por_trama = N // tramas_por_iter

    # Mostrar parámetros principales de forma clara
    print("\n=== Parámetros de canal y simulación ===")
    print(f" Respuesta al impulso del canal h = {h}")
    print(f" Longitud del canal Lh = {Lh}")
    print(f" ValorCP = {ValorCP} -> N_CP usado = {N_CP}")
    print(f" ¿CP suficiente para cubrir Lh-1? -> {N_CP >= max(0, Lh-1)}")
    print(f" Total bits solicitados: {total_bit}")
    print(f" Símbolos OFDM totales (N): {N}")
    print(f" Tramas por iteración = {tramas_por_iter}")
    print(f" Símbolos por trama = {simbolos_por_trama}")
    print(f" Silencio entre tramas (símbolos) = {SimbolosSilencio}")
    print("================================\n")

    # Vector de SNRs (Eb/N0 en dB transformado a Es/N0 para BitsPorSimbolo)
    ebn0_dB_vec = np.arange(0, 11, 2)
    snr_dB_vec = ebn0_dB_vec + 10.0 * np.log10(BitsPorSimbolo)

    runs_per_snr = 10000
    ber_means, ber_stds, all_bers, ultima_info = RunMonteCarlo(
        snr_dB_vec,
        runs=runs_per_snr,
        h=h,
        N_cp=N_CP,
        tramas_por_iter=tramas_por_iter,
        simbolos_silencio=SimbolosSilencio,
        DF_TRUE=DF_TRUE
    )

    # Resultados resumidos BER
    print("\n=== Resultados BER vs Es/N0 y Eb/N0 (AWGN) ===")
    for i, es_db in enumerate(snr_dB_vec):
        eb_db = ebn0_dB_vec[i]
        print(f"Es/N0 = {int(round(es_db))} dB ; Eb/N0 = {int(round(eb_db))} dB -> BER_sim = {ber_means[i]:.3e} +/- {ber_stds[i]:.3e}")
    print()

    # Estadísticas de la última iteración (si existen)
    s = ultima_info.get('stats', {})
    print("=== Estadísticas (última iteración con ruido) ===")
    print(f"Media empírica = {s.get('media_empirica', np.nan)}")
    print(f"Varianza empírica = {s.get('varianza_empirica', np.nan)}")
    print(f"P_signal (última iteración) = {ultima_info.get('PotenciaSenal', np.nan):.6f}")
    print(f"sigma2 (última iteración) = {ultima_info.get('Sigma2', np.nan):.6e}")
    print(f"df_hat (ultima iteración) = {ultima_info.get('df_hat', None)}")
    print(f"phi_hat (ultima iteración) = {ultima_info.get('phi_hat', None)}")
    print(f"P_sum_used (ultima iteración) = {ultima_info.get('P_sum_used', None)}")
    print("====================================\n")

    # Plot BER (mantengo; los diagnósticos por trama ya se imprimieron en consola)
    plt.figure(figsize=(8,5))
    plt.semilogy(ebn0_dB_vec, ber_means, marker='o', linestyle='-', linewidth=2, label='BER simulada (OFDM + AWGN)')
    plt.fill_between(ebn0_dB_vec, np.maximum(ber_means - ber_stds, 1e-12), ber_means + ber_stds, alpha=0.2)
    plt.grid(which='both', linestyle=':', linewidth=0.5)
    plt.xlabel(r'SNR($E_b/N_0$ (dB))')
    plt.ylabel('BER')
    plt.title('BER simulada (OFDM con preámbulo sin CP, canal y AWGN) - estimador CFO coarse')
    plt.ylim(1e-8, 1)
    plt.xlim(ebn0_dB_vec[0], ebn0_dB_vec[-1])
    plt.xticks(list(map(int, np.round(ebn0_dB_vec))))
    plt.legend()
    plt.show()

    # Mostrar tabla y constelación (idéntico a comportamiento anterior)
    TablaYConstelacion()
    Xhat_last = ultima_info.get('Xhat', None)
    if Xhat_last is not None:
        puntos = Xhat_last.reshape(-1)
        plt.figure(figsize=(6,6))
        plt.scatter(np.real(puntos), np.imag(puntos), s=8, alpha=0.6, label='símbolos recibidos (equalizados)')
        qpsk = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        plt.scatter(np.real(qpsk), np.imag(qpsk), s=120, marker='x', linewidths=2, label='const. ideal')
        plt.axhline(0, color='k', linewidth=0.5)
        plt.axvline(0, color='k', linewidth=0.5)
        plt.title('Constelación: símbolos ecualizados (última iteración)')
        plt.xlabel('I'); plt.ylabel('Q')
        plt.gca().set_aspect('equal', 'box')
        plt.grid(True)
        plt.legend()
        plt.show()

    # Comparación final de bits (si hay datos)
    print()
    if ultima_info.get('bits_tx') is not None and ultima_info.get('bits_rx') is not None:
        try:
            coinciden = np.array_equal(ultima_info['bits_tx'][:ultima_info['bits_rx'].size], ultima_info['bits_rx'])
            print("Coinciden todos los bits (última iteración, comparando longitud equivalente)?", coinciden)
        except Exception:
            print("No se pudo comparar bits completos (desajuste de longitudes debido a tramas ignoradas).")
    else:
        print("No hay bits transmitidos/recibidos para comparar (última iteración).")
