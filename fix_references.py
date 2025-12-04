def fix_bridge_references(engine):
    """Fixes mismatched bridge references in the engine.
    
    This solves the issue where training is happening on one bridge instance
    (inside the trainer) but logging/evaluation is looking at a different,
    stale bridge instance (inside the engine).
    """
    print("Checking for bridge reference mismatch...")
    
    trainer_bridge = engine.trainer.bridge
    engine_bridge = engine.bridge
    
    if trainer_bridge is not engine_bridge:
        print(f"⚠️ MISMATCH DETECTED!")
        print(f"Trainer bridge (Training): {id(trainer_bridge)}")
        print(f"Engine bridge (Logging):   {id(engine_bridge)}")
        print("Fixing references to point to the training bridge...")
        
        # Fix engine reference (fixes logging)
        engine.bridge = trainer_bridge
        
        # Fix evaluator references (fixes validation/MMLU)
        if hasattr(engine, 'evaluator'):
            engine.evaluator.bridge = trainer_bridge
            print("Fixed evaluator reference.")
            
        if hasattr(engine, 'mmlu_evaluator'):
            engine.mmlu_evaluator.bridge = trainer_bridge
            print("Fixed MMLU evaluator reference.")
            
        print("✅ References fixed! Your plots should start moving in the next log step.")
    else:
        print("✅ No mismatch found. References are already correct.")
