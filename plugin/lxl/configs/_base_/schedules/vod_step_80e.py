optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

lr_config = dict(
    policy='step',
    warmup=None,
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[35, 45])

momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=80)