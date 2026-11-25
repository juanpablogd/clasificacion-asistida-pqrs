# Fragmento histórico de la celda 16 (índice base cero), solo código fuente.
# No es ejecutable autónomo: necesita el estado de las celdas anteriores.
# Importaciones: Pipeline, TfidfVectorizer, LogisticRegression, StratifiedKFold,
# RandomizedSearchCV, f1_score, accuracy_score y balanced_accuracy_score.
# Variables: SMOKE_MODE, ENABLE_TUNING, SEED, X_train, X_test, y_train, y_test,
# best_text_col, best_f1, best_name y estado del mejor modelo del cuaderno.
# No se incluyen salidas, textos de solicitudes ni objetos de entrenamiento.
if SMOKE_MODE or not ENABLE_TUNING:
    print('Tuning omitido (SMOKE_MODE o ENABLE_TUNING=False).')
else:
    x_train_best = X_train[best_text_col]
    x_test_best = X_test[best_text_col]

    tune_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(strip_accents='unicode', lowercase=True)),
        ('clf', LogisticRegression(class_weight='balanced', random_state=SEED, max_iter=1500, solver='lbfgs', n_jobs=-1))
    ])

    param_dist = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__min_df': [1, 2, 3],
        'tfidf__max_df': [0.85, 0.9, 0.95],
        'tfidf__sublinear_tf': [False, True],
        'clf__C': [0.5, 1.0, 2.0, 4.0, 8.0]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    tuner = RandomizedSearchCV(
        estimator=tune_pipe,
        param_distributions=param_dist,
        n_iter=12,
        scoring='f1_macro',
        cv=cv,
        random_state=SEED,
        n_jobs=-1,
        verbose=0
    )

    tuner.fit(x_train_best, y_train)
    pred_tuned = tuner.predict(x_test_best)

    f1_tuned = f1_score(y_test, pred_tuned, average='macro')
    acc_tuned = accuracy_score(y_test, pred_tuned)
    bal_acc_tuned = balanced_accuracy_score(y_test, pred_tuned)

    print('=== Resultado tuning ligero ===')
    print('Mejores hiperparametros:', tuner.best_params_)
    print(f'F1 macro tuned:          {f1_tuned:.4f}')
    print(f'Accuracy tuned:          {acc_tuned:.4f}')
    print(f'Balanced Accuracy tuned: {bal_acc_tuned:.4f}')

    # Adoptar tuned solo si mejora real en F1
    if f1_tuned > best_f1 + 0.001:
        best_model = tuner.best_estimator_
        best_pred = pred_tuned
        if not best_name.endswith(' + tuning'):
            best_name = f'{best_name} + tuning'
        best_f1 = f1_tuned
        best_acc = acc_tuned
        best_bal_acc = bal_acc_tuned
        print('Se adopta modelo tuneado como mejor modelo final.')
    else:
        print('No se adopta tuning: mejora insuficiente en F1 macro.')
