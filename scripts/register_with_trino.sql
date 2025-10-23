-- Trino Table Registration Script
-- Run this in Trino CLI to register all tables

CALL delta.system.register_table('default', 'listening_history_bronze', '/app/data/bronze/listening_history_bronze');
CALL delta.system.register_table('default', 'my_tracks_features_bronze', '/app/data/bronze/my_tracks_features_bronze');
CALL delta.system.register_table('default', 'my_tracks_features_bronze_synthetic', '/app/data/bronze/my_tracks_features_bronze_synthetic');
CALL delta.system.register_table('default', 'kaggle_tracks_bronze', '/app/data/bronze/kaggle_tracks_bronze');
CALL delta.system.register_table('default', 'listening_with_features', '/app/data/silver/listening_with_features');
CALL delta.system.register_table('default', 'listening_patterns_by_time', '/app/data/gold/descriptive/listening_patterns_by_time');
CALL delta.system.register_table('default', 'top_tracks_by_mood', '/app/data/gold/descriptive/top_tracks_by_mood');
CALL delta.system.register_table('default', 'temporal_trends', '/app/data/gold/descriptive/temporal_trends');
CALL delta.system.register_table('default', 'audio_feature_distributions', '/app/data/gold/descriptive/audio_feature_distributions');
CALL delta.system.register_table('default', 'feature_source_coverage', '/app/data/gold/descriptive/feature_source_coverage');
CALL delta.system.register_table('default', 'mood_time_correlations', '/app/data/gold/diagnostic/mood_time_correlations');
CALL delta.system.register_table('default', 'feature_correlations', '/app/data/gold/diagnostic/feature_correlations');
CALL delta.system.register_table('default', 'weekend_vs_weekday', '/app/data/gold/diagnostic/weekend_vs_weekday');
CALL delta.system.register_table('default', 'mood_shift_patterns', '/app/data/gold/diagnostic/mood_shift_patterns');
CALL delta.system.register_table('default', 'part_of_day_drivers', '/app/data/gold/diagnostic/part_of_day_drivers');
CALL delta.system.register_table('default', 'mood_predictions', '/app/data/gold/predictive/mood_predictions');
CALL delta.system.register_table('default', 'energy_forecasts', '/app/data/gold/predictive/energy_forecasts');
CALL delta.system.register_table('default', 'mood_classifications', '/app/data/gold/predictive/mood_classifications');
CALL delta.system.register_table('default', 'model_performance_metrics', '/app/data/gold/predictive/model_performance_metrics');
