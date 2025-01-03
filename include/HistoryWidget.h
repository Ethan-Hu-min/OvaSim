#ifndef HISTORYWIDGET_H
#define HISTORYWIDGET_H

#include "ui_HistoryWidget.h"

#include <QWidget>

namespace Ui {
class HistoryWidget;
}

class HistoryWidget : public QWidget
{
    Q_OBJECT

public:
    explicit HistoryWidget(QWidget *parent = nullptr);
    ~HistoryWidget();
    void closeEvent(QCloseEvent*);

signals:
    void ExitWin();
private:
    Ui::HistoryWidget ui;
};

#endif // HISTORYWIDGET_H
