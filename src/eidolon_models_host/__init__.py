"""Facts about the Host a model service runs on, not about any one model.

Here because two services need the same answer and neither should have to
import the other to get it: the core allocation belongs to the board, and
`eidolon_models_asr` importing it from `eidolon_models_tts` (or the reverse)
would make one service's presence a condition of the other's.
"""
