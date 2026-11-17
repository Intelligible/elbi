A `.elbi/lifecycle.yaml` or a stored agent-authored derivation under `.elbi/authored/`
that cannot be read, is not UTF-8, or is not valid YAML is now reported as a
configuration error. Commands that load the project, such as `elbi validate` and
`elbi test`, print an error line for these files, where before they printed a traceback.
