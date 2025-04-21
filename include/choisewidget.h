#ifndef CHOISEWIDGET_H
#define CHOISEWIDGET_H

#include <QWidget>
#include "ui_choisewidget.h"
namespace Ui {
class ChoiseWidget;
}

class ChoiseWidget : public QWidget
{
    Q_OBJECT


public:
    explicit ChoiseWidget(QWidget *parent = nullptr);
    ~ChoiseWidget();
    void getInfo(QString& name, QString& mode, QString& sample);

signals:    void ExitWin();

private slots:
    void on_startbutton_clicked();



private:
    Ui::ChoiseWidget *ui;
};

#endif // CHOISEWIDGET_H
