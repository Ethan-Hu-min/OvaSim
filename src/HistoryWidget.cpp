#include "HistoryWidget.h"


HistoryWidget::HistoryWidget(QWidget *parent)
    : QWidget(parent)
{
    setAttribute(Qt::WA_DeleteOnClose);
    ui.setupUi(this);
}

HistoryWidget::~HistoryWidget()
{
}

void HistoryWidget::closeEvent(QCloseEvent*) {
    emit ExitWin();
}