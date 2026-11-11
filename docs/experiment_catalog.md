# Каталог экспериментов

Для прежних серий сохранены 114 конфигураций и протоколов и 890 записей метрик. [Конфигурации](../configs/experiment_history.json) · [Сводная таблица](../results/metric_index.csv)

| Опыт | Вопрос | Результаты |
| --- | --- | --- |
| `data_quality_20261104` | Есть ли пропуски, дубликаты и неполные сутки? | [Таблица](../research/01_data_and_protocol/results/data_quality_20261104__processed_quality.csv) |
| `initial_pilot` | Сколько времени и памяти требует пилот? | [Сводная таблица](../results/metric_index.csv) |
| `active_history` | Помогает ли двухбашенной модели своя история пользователя? | [Сводная таблица](../results/metric_index.csv) |
| `expanded_als_20261105` | Повышает ли качество ALS расширение аудитории? | [Таблица](../research/02_baselines_and_classical/results/expanded_als_20261105__scale_comparison.csv) |
| `history_20261103` | Что меняется, если оставить в истории только сильные действия? | [Сводная таблица](../results/metric_index.csv) |
| `main_50000` | Как меняются метрики при увеличении выборки? | [Сводная таблица](../results/metric_index.csv) |
| `profile_audit_20261104` | Что даёт полный профиль ALS при тех же товарных факторах? | [Таблица](../research/02_baselines_and_classical/results/profile_audit_20261104__comparison.csv) |
| `strong_admission_als_20261105` | Помогает ли отбор дополнительных пользователей по сильным действиям? | [Таблица](../research/02_baselines_and_classical/results/strong_admission_als_20261105__population_comparison.csv) |
| `temporal_als_20261104` | Как эффект квоты ALS зависит от недели и seed? | [Таблица](../research/05_fusion_and_robustness/results/temporal_als_20261104__window_summary.csv) |
| `candidate_calibration_20261104` | Как размер каталога влияет на время обучения и поиска? | [Сводная таблица](../results/metric_index.csv) |
| `catalog20000_20261104` | Как работают две башни на каталоге из 20 000 товаров? | [Сводная таблица](../results/metric_index.csv) |
| `ccl_20261108` | Как два варианта CCL меняют Recall двухбашенной модели? | [Таблица](../research/03_two_tower/results/ccl_20261108__stopped_study_summary.csv) |
| `expansion_20261103` | Сколько примеров модель получает при коротком обучении? | [Сводная таблица](../results/metric_index.csv) |
| `followup_20261103` | Как негативные примеры и веса действий влияют на качество башен? | [Сводная таблица](../results/metric_index.csv) |
| `full_catalog_20261104` | Что даёт переход с четырёх недель обучения на восемь? | [Таблица](../research/03_two_tower/results/full_catalog_20261104__comparison.csv) |
| `interests_20261104` | Помогают ли знакомые категории находить новые товары? | [Таблица](../research/05_fusion_and_robustness/results/interests_20261104__comparison.csv) |
| `investigation_20261104` | Как меняется Recall с полным softmax и выборкой негативов? | [Таблица](../research/03_two_tower/results/investigation_20261104__loss_comparison.csv) |
| `large_mixed_20261105` | Как влияют состав истории, признаки товаров и расширение аудитории? | [Таблица](../research/03_two_tower/results/large_mixed_20261105__completed_comparison.csv) |
| `mixed_history_20261104` | Сколько целей сохраняется в смешанной истории с view? | [Таблица](../research/03_two_tower/results/mixed_history_20261104__evaluation__input_coverage.csv) |
| `multi_interest_20261105` | Лучше ли кластеры истории случайного разбиения? | [Таблица](../research/05_fusion_and_robustness/results/multi_interest_20261105__comparison.csv) |
| `stability_lr_20261108` | Как скорость обучения и расчёт градиентов влияют на длинный запуск? | [Таблица](../research/03_two_tower/results/stability_lr_20261108__comparison.csv) |
| `training_population_20261105` | Какие будущие товары представлены в обучающих позитивных примерах? | [Сводная таблица](../results/metric_index.csv) |
| `training_regime_20261108` | На каком этапе длительного обучения возникает нестабильность? | [Сводная таблица](../results/metric_index.csv) |
| `user_week_balance_20261104` | Помогают ли равные веса пользователей по неделям? | [Сводная таблица](../results/metric_index.csv) |
| `prefix_sequence_20261104` | Помогает ли внимание по сравнению с усреднением истории? | [Таблица](../research/04_sequence_models/results/prefix_sequence_20261104__comparison.csv) |
| `channel_contribution_20261104` | Сколько целей добавляет и вытесняет ALS при фиксированной длине выдачи? | [Таблица](../research/05_fusion_and_robustness/results/channel_contribution_20261104__comparison.csv) |
| `focused_hypotheses_20261107` | Как влияют агрегация ALS, давность личной частоты и температура башен? | [Описание](../research/05_fusion_and_robustness/README.md) |
| `repeat_cold_channels_20261105` | Какие каналы помогают повторным и холодным товарам? | [Таблица](../research/05_fusion_and_robustness/results/repeat_cold_channels_20261105__cold_temporal__comparison.csv) |
| `windows_20261103` | Сохраняется ли эффект объединения на разных неделях? | [Таблица](../research/05_fusion_and_robustness/results/temporal_als_20261104__window_summary.csv) |
| `closing_checks_20261107` | Как зафиксированный рецепт работает на отложенной неделе? | [Таблица](../research/06_final_results/results/closing_checks_20261107__final_comparison.csv) |
| `best_models_20261110` | Какие методы и объединения лучше на общем периоде 1293–1299? | [Описание](../research/07_best_models/README.md) |
