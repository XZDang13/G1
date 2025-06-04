import gymnasium

gymnasium.register(
    id="G1Walk-v0",
    entry_point=f"{__name__}.walk_env:G1WalkEnv"
)