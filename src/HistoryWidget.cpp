#include "HistoryWidget.h"


HistoryWidget::HistoryWidget(QWidget *parent)
    : QWidget(parent)
{
    setAttribute(Qt::WA_DeleteOnClose);
    ui.setupUi(this);
    ui.tableWidget->setRowCount(10);
    ui.tableWidget->setColumnCount(5);
    ui.tableWidget->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));
    QStringList headers;
    headers << "用户名" << "操作时间" << "操作时长" << "获卵数" << "操作警告";
    ui.tableWidget->setHorizontalHeaderLabels(headers);
    ui.tableWidget->setItem(0, 0, new QTableWidgetItem("张三"));
    ui.tableWidget->setItem(0, 1, new QTableWidgetItem("2025年03月12日15时35分"));
    ui.tableWidget->setItem(0, 2, new QTableWidgetItem("5分44秒"));
    ui.tableWidget->setItem(0, 3, new QTableWidgetItem("12"));
    ui.tableWidget->setItem(0, 4, new QTableWidgetItem("错误穿刺膀胱"));
    ui.tableWidget->setItem(0, 0, new QTableWidgetItem("张三"));
    ui.tableWidget->resizeColumnsToContents();
    ui.tableWidget->resizeRowsToContents();

}

HistoryWidget::~HistoryWidget()
{
}

void HistoryWidget::closeEvent(QCloseEvent*) {
    emit ExitWin();
}