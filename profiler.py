from ydata_profiling import ProfileReport


def create_profile_report(data, title, report_name):
    report = ProfileReport(data, title=title, correlations={
        "pearson": {"calculate": True},
        "spearman": {"calculate": True},
        "kendall": {"calculate": True}
    })
    report.to_file(report_name + ".html")
