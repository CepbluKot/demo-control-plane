{{/*
Общий префикс для имён ресурсов.
*/}}
{{- define "log-summarizer.fullname" -}}
{{- printf "log-summarizer" }}
{{- end }}

{{/*
Имя Secret с credentials.
*/}}
{{- define "log-summarizer.secretName" -}}
{{- printf "log-summarizer-credentials" }}
{{- end }}

{{/*
Имя PVC с данными.
*/}}
{{- define "log-summarizer.dataPVC" -}}
{{- printf "log-summarizer-data" }}
{{- end }}

{{/*
Имя PVC с артефактами.
*/}}
{{- define "log-summarizer.runsPVC" -}}
{{- printf "log-summarizer-runs" }}
{{- end }}

{{/*
Общие labels.
*/}}
{{- define "log-summarizer.labels" -}}
app.kubernetes.io/name: log-summarizer
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}
